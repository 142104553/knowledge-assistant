#!/usr/bin/env python3
"""
Chroma → Milvus 数据迁移脚本

用法：
    python migrate_chroma_to_milvus.py

功能：
    1. 从 ./chroma_db 读取所有 chunk（文本 + 向量 + 元数据）
    2. 写入 Milvus（默认 Milvus Lite 本地文件模式）
    3. 对比数量，验证数据完整性

环境变量（可选）：
    CHROMA_PERSIST_DIR=./chroma_db     Chroma 数据目录
    MILVUS_URI=./milvus.db             Milvus 连接地址（文件或 http）
    COLLECTION_NAME=documents          Collection 名称
"""

import os
import sys
from typing import List, Dict, Any


def main():
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    milvus_uri = os.environ.get("MILVUS_URI", "./milvus.db")
    collection_name = os.environ.get("COLLECTION_NAME", "documents")

    print("=" * 60)
    print("Chroma to Milvus Migration")
    print("=" * 60)
    print(f"Source (Chroma): {chroma_dir}")
    print(f"Target (Milvus): {milvus_uri}")
    print(f"Collection: {collection_name}")
    print()

    # ── 1. 连接 Chroma ──
    print("[1/4] Connecting Chroma...")
    try:
        import chromadb
    except ImportError:
        print("[FAIL] Please install Chroma: pip install chromadb")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    try:
        chroma_coll = chroma_client.get_collection(collection_name)
    except Exception as e:
        print(f"[FAIL] Chroma collection '{collection_name}' not found: {e}")
        sys.exit(1)

    chroma_count = chroma_coll.count()
    print(f"[OK] Chroma connected, total {chroma_count} chunks")

    if chroma_count == 0:
        print("[WARN] No data in Chroma, nothing to migrate")
        sys.exit(0)

    # ── 2. 读取所有数据 ──
    print("[2/4] Reading Chroma data...")
    results = chroma_coll.get(include=["documents", "embeddings", "metadatas"])

    ids: List[str] = results.get("ids", [])
    documents: List[str] = results.get("documents", [])
    embeddings: List[List[float]] = results.get("embeddings", [])
    metadatas: List[Dict[str, Any]] = results.get("metadatas", [])

    print(f"    IDs: {len(ids)}, Docs: {len(documents)}, Embeddings: {len(embeddings)}, Metas: {len(metadatas)}")

    if len(documents) != chroma_count:
        print(f"[WARN] Document count mismatch: read={len(documents)}, count={chroma_count}")

    # ── 3. 连接 Milvus ──
    print("[3/4] Connecting Milvus...")
    try:
        from vectorstore.milvus_store import MilvusVectorStore
    except ImportError:
        print("[FAIL] Please install Milvus deps: pip install pymilvus milvus-lite")
        sys.exit(1)

    dim = len(embeddings[0]) if len(embeddings) > 0 else 512
    milvus = MilvusVectorStore(
        collection_name=collection_name,
        dimension=dim,
        uri=milvus_uri,
    )
    print(f"[OK] Milvus connected (dim={dim})")

    existing = milvus.count()
    if existing > 0:
        print(f"[WARN] Milvus already has {existing} records")
        confirm = input("Clear and re-migrate? [y/N]: ").strip().lower()
        if confirm == "y":
            milvus.clear()
            print("[OK] Milvus cleared")
        else:
            print("[ABORT] Migration cancelled")
            sys.exit(0)

    # ── 4. 分批写入 ──
    print("[4/4] Writing to Milvus...")
    batch_size = 100
    total = len(documents)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        milvus.add_texts(
            texts=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        progress = (end / total) * 100
        print(f"    Progress: {end}/{total} ({progress:.1f}%)")

    # Verify
    milvus_count = milvus.count()
    print()
    print("=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print(f"Source (Chroma):  {chroma_count}")
    print(f"Target (Milvus):  {milvus_count}")
    if chroma_count == milvus_count:
        print("[OK] Count matches, migration successful")
    else:
        print(f"[WARN] Count mismatch: diff={abs(chroma_count - milvus_count)}")
    print()

    # Sample verification
    print("[Verify] Sample search test...")
    sample_emb = embeddings[0]
    results = milvus.similarity_search(sample_emb, top_k=3)
    print(f"    Search with embedding[0], returned {len(results)} results")
    for j, r in enumerate(results[:2]):
        print(f"      [{j}] {r.content[:50]}...")
    print("[OK] Sample verification passed")

    # Next steps
    print()
    print("Next steps:")
    print("   1. Edit .env: VECTORSTORE_PROVIDER=milvus")
    print(f"   2. Set MILVUS_URI={milvus_uri}")
    print("   3. Restart backend service")
    print("   4. Optional: remove Chroma data: rm -rf ./chroma_db")


if __name__ == "__main__":
    main()
