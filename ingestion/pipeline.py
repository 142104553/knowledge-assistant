"""
第9课（Part 1）：端到端文档摄取管道

把整个 ingestion 流程串联：
    文件 → 加载 → 分块 → Embedding → 存入向量库

使用方式：
    # 摄取单个文件
    python -m ingestion.pipeline --file ./docs/manual.pdf
    
    # 摄取整个目录
    python -m ingestion.pipeline --dir ./docs/
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from app.core.config import get_settings
from models.document import DocumentChunk


def run_ingestion_pipeline(
    input_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    doc_id: Optional[str] = None
) -> List[DocumentChunk]:
    """
    执行完整的文档摄取流程

    Args:
        input_path: 文件或目录路径
        chunk_size: 分块大小（默认从配置读取）
        chunk_overlap: 重叠大小（默认从配置读取）

    Returns:
        成功摄取的所有 chunk 列表
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # 1. 确定文件列表
    path = Path(input_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        supported_exts = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}
        files = [f for f in path.rglob("*") if f.suffix.lower() in supported_exts]
        print(f"📁 发现 {len(files)} 个待处理文件")
    else:
        raise ValueError(f"无效路径: {input_path}")

    if not files:
        print("⚠️ 没有找到支持的文件")
        return []

    # 2. 初始化组件
    from ingestion.loaders.factory import DocumentLoaderFactory
    from ingestion.splitters.factory import get_splitter
    from embeddings.factory import EmbeddingFactory
    from vectorstore.factory import VectorStoreFactory

    # 使用配置中的 Embedding 客户端（BGE 本地模型）
    embedder = EmbeddingFactory.create(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model
    )
    vector_store = VectorStoreFactory.create(
        provider=settings.vectorstore_provider,
        collection_name=settings.vectorstore_collection,
        dimension=embedder.dimension,
        persist_directory=settings.chroma_persist_dir
    )

    # 3. 处理每个文件
    all_chunks = []
    for file_path in files:
        print(f"\n📄 处理: {file_path.name}")

        try:
            # 3.1 加载
            loader = DocumentLoaderFactory.get_loader(str(file_path))
            raw_chunks = loader.load(str(file_path))
            print(f"   ✓ 加载完成: {len(raw_chunks)} 页/段")

            # 3.2 分块
            # 根据文件类型选择分块策略
            strategy = "markdown" if file_path.suffix.lower() in {".md", ".markdown"} else "recursive"
            splitter = get_splitter(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_chunks = splitter.split(raw_chunks)
            print(f"   ✓ 分块完成: {len(split_chunks)} 个 chunk")

            # 注入 doc_id 到 metadata
            if doc_id:
                for chunk in split_chunks:
                    chunk.metadata["doc_id"] = doc_id

            all_chunks.extend(split_chunks)

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue

    if not all_chunks:
        print("\n⚠️ 没有成功提取任何文本块")
        return []

    # 4. Embedding 向量化（批量处理）
    print(f"\n🔢 开始 Embedding 向量化 ({embedder.model_name}, {embedder.dimension}D)...")

    batch_size = 100  # 每批处理 100 个 chunk
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c.content for c in batch]
        metadatas = [c.metadata for c in batch]

        embeddings = embedder.embed(texts)

        # 存入向量库
        vector_store.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"   批次 {i // batch_size + 1}/{total_batches} 完成 ({len(batch)} chunks)")

    print(f"\n✅ 摄取完成！总计: {len(all_chunks)} 个 chunk")
    print(f"   向量库当前文档数: {vector_store.count()}")

    return all_chunks


def main():
    """命令行入口（自动注入 doc_id，与 API 行为一致）"""
    import hashlib
    parser = argparse.ArgumentParser(description="文档摄取管道")
    parser.add_argument("--file", help="单个文件路径")
    parser.add_argument("--dir", help="目录路径（递归处理所有支持的文件）")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="块间重叠")

    args = parser.parse_args()

    if args.file:
        # 自动计算 MD5 作为 doc_id
        hasher = hashlib.md5()
        with open(args.file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        doc_id = hasher.hexdigest()
        run_ingestion_pipeline(args.file, args.chunk_size, args.chunk_overlap, doc_id=doc_id)
    elif args.dir:
        # 目录模式下每个文件单独计算 doc_id
        from pathlib import Path
        for ext in {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}:
            for file_path in Path(args.dir).rglob(f"*{ext}"):
                hasher = hashlib.md5()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                doc_id = hasher.hexdigest()
                run_ingestion_pipeline(str(file_path), args.chunk_size, args.chunk_overlap, doc_id=doc_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
