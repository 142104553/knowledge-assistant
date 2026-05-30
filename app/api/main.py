"""
FastAPI RESTful API 服务（v0.2）

功能：
- 文件上传（multipart/form-data）
- 对话历史持久化（SQLite）
- 文档去重/覆盖更新（MD5 哈希）
- 删除文档
"""

from dotenv import load_dotenv
load_dotenv()

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings, Settings
from models.document import QueryRequest, ChatResponse
from models.database import init_db, save_message, get_conversation_history, list_documents, delete_document_meta

# 全局依赖（lifespan 中初始化）
rag_chain = None
agent_router = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, agent_router, vector_store

    # 初始化数据库
    init_db()

    settings = get_settings()
    print(f"🚀 启动 {settings.app_name} v{settings.app_version}")
    print(f"📦 Embedding: {settings.embedding_provider} / {settings.embedding_model}")

    # 初始化 Embedding
    from embeddings.factory import EmbeddingFactory
    embedder = EmbeddingFactory.create(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model
    )

    # 初始化向量库
    from vectorstore.factory import VectorStoreFactory
    chroma_path = Path(settings.chroma_persist_dir)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        print(f"[注意] 检测到已有向量库: {settings.chroma_persist_dir}")
        print(f"[注意] 当前 Embedding 维度: {embedder.dimension}D")

    vector_store = VectorStoreFactory.create(
        provider=settings.vectorstore_provider,
        collection_name=settings.vectorstore_collection,
        dimension=embedder.dimension,
        persist_directory=settings.chroma_persist_dir
    )

    # 初始化检索链
    from rag.retrievers.hybrid import HybridRetriever
    from rag.post_processors.reranker import NoOpReranker
    from rag.chains.rag_chain import LLMClient, RAGChain
    from agent.router import AgentRouter

    retriever = HybridRetriever(vector_store=vector_store)
    reranker = NoOpReranker()
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model
    )
    rag_chain = RAGChain(
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        max_context_tokens=settings.max_context_tokens
    )
    agent_router = AgentRouter(llm=llm, rag_chain=rag_chain)

    print(f"✅ 初始化完成 | 向量库文档数: {vector_store.count()}")

    yield

    print("👋 应用关闭")


app = FastAPI(
    title="企业智能知识助手 API",
    description="基于 RAG + Agent 架构的私有知识库问答系统",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# 健康检查
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "enterprise-knowledge-assistant", "version": "0.2.0"}


# ═══════════════════════════════════════════════════════════
# 问答接口
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: QueryRequest):
    """问答接口（主入口），返回后自动保存到对话历史"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    settings = get_settings()

    # 保存用户提问
    if request.session_id:
        save_message(request.session_id, "user", request.query)

    try:
        if settings.enable_agent and request.enable_agent:
            intent = agent_router.analyze_intent(request.query)
            plan = agent_router.plan(intent, request.query)
            response = agent_router.execute(plan, request.query)
        else:
            response = rag_chain.invoke(request)

        # 保存助手回答
        if request.session_id:
            sources = [{"content": s.content, "metadata": s.metadata, "score": s.score} for s in response.sources]
            save_message(request.session_id, "assistant", response.answer, sources)

        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部服务器错误，请稍后重试")


@app.get("/api/v1/chat/history")
def get_history(session_id: str = Query(..., description="对话 session ID")):
    """获取某个 session 的对话历史"""
    return {"session_id": session_id, "messages": get_conversation_history(session_id)}


@app.delete("/api/v1/chat/history")
def clear_history(session_id: str = Query(..., description="对话 session ID")):
    """清空某个 session 的对话历史"""
    from models.database import clear_conversation
    clear_conversation(session_id)
    return {"status": "success", "message": f"已清空 session {session_id} 的对话历史"}


# ═══════════════════════════════════════════════════════════
# 文档摄取接口（真正的文件上传）
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/ingest")
def ingest_document(file: UploadFile = File(...)):
    """
    文档摄取接口（v0.2）

    - 接收上传的文件（multipart/form-data）
    - 计算 MD5 作为 doc_id，实现去重和覆盖
    - 如果同名/同内容文件已存在，先删除旧数据再入库
    """
    from ingestion.pipeline import run_ingestion_pipeline
    from models.database import save_document_meta, get_document_meta, delete_document_meta

    settings = get_settings()

    # 保存到临时文件
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 计算文件 MD5 作为 doc_id
        hasher = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        doc_id = hasher.hexdigest()

        # 检查是否已存在相同文件
        existing = get_document_meta(doc_id)
        if existing:
            print(f"[去重] 检测到已上传的相同文件: {filename} (doc_id={doc_id[:8]}...)")
            print(f"[去重] 先删除旧数据...")
            # 按 doc_id 删除向量库中的旧 chunk
            if vector_store:
                vector_store.delete(filter_dict={"doc_id": doc_id})
            delete_document_meta(doc_id)

        # 执行摄取（把 doc_id 注入到 metadata 中）
        chunks = run_ingestion_pipeline(tmp_path, doc_id=doc_id)

        # 记录文档元数据
        save_document_meta(doc_id=doc_id, filename=file.filename, chunk_count=len(chunks))

        return {
            "status": "success",
            "filename": file.filename,
            "doc_id": doc_id,
            "chunks_ingested": len(chunks),
            "is_update": existing is not None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════
# 文档管理接口
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/documents")
def list_docs():
    """列出所有已上传的文档"""
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str):
    """删除指定文档（从向量库和元数据中）"""
    try:
        if vector_store:
            vector_store.delete(filter_dict={"doc_id": doc_id})
        delete_document_meta(doc_id)
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# 统计接口
# ═══════════════════════════════════════════════════════════

@app.get("/api/v1/stats")
def get_stats():
    """获取知识库统计信息"""
    settings = get_settings()
    docs = list_documents()

    return {
        "total_documents_in_store": vector_store.count() if vector_store else 0,
        "total_files_uploaded": len(docs),
        "collection_name": settings.vectorstore_collection,
        "embedding_model": settings.embedding_model,
        "vectorstore_provider": settings.vectorstore_provider
    }
