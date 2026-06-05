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

import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from app.core.config import get_settings, Settings
from models.document import QueryRequest, ChatResponse
from models.database import init_db, save_message, get_conversation_history, list_documents, delete_document_meta, clear_all_data

# 全局依赖（lifespan 中初始化）
rag_chain = None
agent_router = None
vector_store = None
bm25_retriever = None
retriever = None


def _rebuild_bm25():
    """重建 BM25 语料库（文档增删后调用）"""
    global bm25_retriever, retriever, vector_store
    if not vector_store:
        return False
    try:
        from rag.retrievers.hybrid import BM25Retriever
        all_docs = vector_store.get_all()
        if not all_docs:
            bm25_retriever = None
            if retriever:
                retriever.bm25_retriever = None
            print("[OK] BM25 cleared (no docs)")
            return True
        new_bm25 = BM25Retriever(
            texts=[d.content for d in all_docs],
            metadatas=[d.metadata for d in all_docs]
        )
        bm25_retriever = new_bm25
        if retriever:
            retriever.bm25_retriever = new_bm25
        print(f"[OK] BM25 rebuilt | docs: {len(all_docs)}")
        return True
    except Exception as e:
        print(f"[WARN] BM25 rebuild failed: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, agent_router, vector_store, bm25_retriever, retriever

    # 初始化数据库
    init_db()

    settings = get_settings()
    print(f"[START] {settings.app_name} v{settings.app_version}")
    print(f"[EMBED] {settings.embedding_provider} / {settings.embedding_model}")

    # 初始化 Embedding
    from embeddings.factory import EmbeddingFactory
    embedder = EmbeddingFactory.create(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model
    )

    # 初始化向量库
    from vectorstore.factory import VectorStoreFactory
    vs_provider = settings.vectorstore_provider.lower()

    vs_kwargs = {}
    if vs_provider == "chroma":
        chroma_path = Path(settings.chroma_persist_dir)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            print(f"[INFO] Vector store exists: {settings.chroma_persist_dir}")
        vs_kwargs["persist_directory"] = settings.chroma_persist_dir
    elif vs_provider == "milvus":
        vs_kwargs = {
            "host": settings.milvus_host,
            "port": settings.milvus_port,
        }
        if settings.milvus_uri:
            vs_kwargs["uri"] = settings.milvus_uri
        if settings.milvus_token:
            vs_kwargs["token"] = settings.milvus_token
        print(f"[INFO] Connecting to Milvus: {settings.milvus_host}:{settings.milvus_port}")

    print(f"[INFO] Embedding dim: {embedder.dimension}D")

    vector_store = VectorStoreFactory.create(
        provider=vs_provider,
        collection_name=settings.vectorstore_collection,
        dimension=embedder.dimension,
        **vs_kwargs
    )

    # 初始化检索链
    from rag.retrievers.hybrid import HybridRetriever, BM25Retriever
    from rag.post_processors.reranker import CrossEncoderReranker
    from rag.chains.rag_chain import RAGChain
    from rag.llm.factory import create_llm_client, get_provider_info
    from agent.router import AgentRouter
    from agent.langgraph_router import LangGraphAgentRouter

    # 构建 BM25 语料库（从向量库读取已有文档）
    doc_count = vector_store.count()
    if doc_count > 0:
        try:
            all_docs = vector_store.get_all()
            bm25_retriever = BM25Retriever(
                texts=[d.content for d in all_docs],
                metadatas=[d.metadata for d in all_docs]
            )
            print(f"[OK] BM25 corpus built | docs: {len(all_docs)}")
        except Exception as e:
            print(f"[WARN] BM25 build failed: {e}, fallback to dense only")

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever
    )

    # 启用 Cross-Encoder Reranker（轻量版）
    from rag.post_processors.reranker import NoOpReranker
    try:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            local_path=settings.reranker_local_path
        )
        print(f"[OK] Cross-Encoder Reranker loaded")
    except Exception as e:
        print(f"[WARN] Cross-Encoder load failed: {e}, using NoOpReranker")
        reranker = NoOpReranker()

    llm = create_llm_client(
        client_type=settings.llm_client_type,
        provider=settings.llm_default_provider,
        settings=settings
    )
    provider_info = get_provider_info(settings)
    print(f"[OK] {settings.llm_client_type.upper()} LLM client enabled | provider={provider_info['provider']} model={provider_info['model']}")
    rag_chain = RAGChain(
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        max_context_tokens=settings.max_context_tokens,
        answer_status_threshold_high=settings.answer_status_threshold_high,
        answer_status_threshold_low=settings.answer_status_threshold_low,
    )

    if settings.agent_router_type == "langgraph":
        from agent.tools.base import DEFAULT_TOOLS
        agent_router = LangGraphAgentRouter(
            llm=llm,
            rag_chain=rag_chain,
            tools=DEFAULT_TOOLS
        )
        print("[OK] LangGraph AgentRouter enabled")
    else:
        agent_router = AgentRouter(llm=llm, rag_chain=rag_chain)
        print("[OK] Legacy AgentRouter enabled")

    print(f"[OK] Init done | docs: {doc_count} | hybrid: {'Y' if bm25_retriever else 'N'} | rerank: {'Y' if not isinstance(reranker, NoOpReranker) else 'N'}")

    yield

    print("[SHUTDOWN] App stopped")


app = FastAPI(
    title="企业智能知识助手 API",
    description="基于 RAG + Agent 架构的私有知识库问答系统",
    version="0.2.0",
    lifespan=lifespan
)

# CORS：从配置读取允许的来源，生产环境不应开放 *
_settings = get_settings()
_cors_origins = [o.strip() for o in (_settings.cors_origins or "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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


@app.post("/api/v1/chat/stream")
def chat_stream(request: QueryRequest):
    """流式问答接口，返回 SSE 事件流"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    settings = get_settings()

    def event_generator():
        # 保存用户提问
        if request.session_id:
            save_message(request.session_id, "user", request.query)

        full_answer = ""
        try:
            if settings.enable_agent and request.enable_agent:
                # Agent 模式暂不支持流式，fallback 到非流式
                intent = agent_router.analyze_intent(request.query)
                plan = agent_router.plan(intent, request.query)
                response = agent_router.execute(plan, request.query)
                full_answer = response.answer
                yield f"data: {json.dumps({'type': 'token', 'content': full_answer})}\n\n"
                sources = response.sources
            else:
                for chunk in rag_chain.stream(request):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                sources = rag_chain.get_last_sources()
                answer_status = rag_chain.get_last_answer_status()

            # 发送 sources（RAG 模式附带 answer_status）
            sources_json = [
                {"content": s.content, "metadata": s.metadata, "score": s.score}
                for s in sources
            ]
            if not (settings.enable_agent and request.enable_agent):
                # RAG 模式：附带 answer_status
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_json, 'answer_status': answer_status})}\n\n"
            else:
                # Agent 模式：暂无 answer_status
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_json})}\n\n"

            # 保存助手回答
            if request.session_id:
                save_message(request.session_id, "assistant", full_answer, sources_json)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


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
            print(f"[DEDUP] Same file detected: {filename} (doc_id={doc_id[:8]}...)")
            print(f"[去重] 先删除旧数据...")
            # 按 doc_id 删除向量库中的旧 chunk
            if vector_store:
                vector_store.delete(filter_dict={"doc_id": doc_id})
            delete_document_meta(doc_id)

        # 执行摄取（把 doc_id 注入到 metadata 中）
        chunks = run_ingestion_pipeline(tmp_path, doc_id=doc_id)

        # 记录文档元数据
        save_document_meta(doc_id=doc_id, filename=file.filename, chunk_count=len(chunks))

        # 重建 BM25 语料库（新增文档后热更新）
        _rebuild_bm25()

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

        # 重建 BM25 语料库（删除文档后热更新）
        _rebuild_bm25()

        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# 统计接口
# ═══════════════════════════════════════════════════════════

@app.post("/api/v1/database/clear")
def clear_database():
    """
    一键清空数据库（向量库 + 文档元数据 + 对话历史）

    ⚠️ 危险操作，不可恢复！
    """
    global bm25_retriever, retriever
    try:
        # 1. 清空向量库
        if vector_store:
            vector_store.clear()
            print("[OK] Vector store cleared")

        # 2. 清空 SQLite（documents + conversations）
        clear_all_data()
        print("[OK] SQLite cleared")

        # 3. 重置 BM25
        bm25_retriever = None
        if retriever:
            retriever.bm25_retriever = None
        print("[OK] BM25 reset")

        return {"status": "success", "message": "数据库已清空"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
def get_stats():
    """获取知识库统计信息"""
    settings = get_settings()
    docs = list_documents()

    return {
        "total_documents_in_store": vector_store.count() if vector_store else 0,
        "total_files_uploaded": len(docs),        "collection_name": settings.vectorstore_collection,
        "embedding_model": settings.embedding_model,
        "vectorstore_provider": settings.vectorstore_provider
    }
