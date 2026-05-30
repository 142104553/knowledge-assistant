"""
第5课：向量数据库封装

核心接口：
    store = VectorStoreFactory.create("chroma", collection_name="docs", dimension=1536)
    
    # 添加文档
    store.add_texts(
        texts=["文本1", "文本2"],
        embeddings=[[0.1, ...], [0.2, ...]],
        metadatas=[{"file": "a.pdf"}, {"file": "b.pdf"}]
    )
    
    # 检索
    results = store.similarity_search(query_vector=[0.1, ...], top_k=5, filter={"file": "a.pdf"})
    # 返回: List[RetrievedChunk]

设计要点：
- 统一的 add / search / delete 接口，底层可以是 Chroma/Qdrant/Milvus
- 元数据过滤：支持按文件名、文档类型等字段过滤检索范围
- ID 管理：每个 chunk 有唯一 ID，支持增量更新和删除
"""

import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from models.document import DocumentChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """
    向量数据库抽象基类
    
    所有实现必须支持：
    - add_texts: 批量插入文本、向量、元数据
    - similarity_search: 向量相似度检索
    - delete: 按 ID 或过滤条件删除
    """

    def __init__(self, collection_name: str, dimension: int):
        self.collection_name = collection_name
        self.dimension = dimension

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        批量添加文档到向量数据库
        
        Args:
            texts: 原始文本内容列表
            embeddings: 对应的向量列表
            metadatas: 每条记录的元数据（如文件名、页码）
            ids: 自定义 ID，不提供则自动生成 UUID
            
        Returns:
            实际插入的 ID 列表
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        向量相似度检索
        
        Args:
            query_embedding: 查询文本的向量
            top_k: 返回最相近的 K 个结果
            filter_dict: 元数据过滤条件，如 {"doc_type": "pdf"}
            score_threshold: 相似度阈值，低于此值的结果被过滤
            
        Returns:
            检索到的 chunk 列表，按相似度从高到低排序
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        删除文档
        
        Args:
            ids: 按 ID 列表删除
            filter_dict: 按元数据条件批量删除（如删除某个文件的所有 chunk）
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """返回 Collection 中的文档总数"""
        pass

    def _generate_ids(self, n: int) -> List[str]:
        """生成 UUID 作为文档 ID"""
        return [str(uuid.uuid4()) for _ in range(n)]


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量数据库（本地开发首选）
    
    特点：
    - 纯 Python，pip install 即可用
    - 数据默认持久化到本地磁盘（./chroma_db）
    - 自动处理 Embedding 存储和索引
    
    安装：pip install chromadb
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine"
    ):
        super().__init__(collection_name, dimension)
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")

        self.persist_directory = persist_directory or "./chroma_db"
        
        # 创建/连接本地 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False  # 关闭匿名数据收集
            )
        )

        # 获取或创建 Collection
        # metadata 中的 "hnsw:space" 指定距离度量方式：cosine / l2 / ip
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        if not texts:
            return []

        if ids is None:
            ids = self._generate_ids(len(texts))

        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Chroma 的 add 接口
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return ids

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        # Chroma 的 query 接口
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,  # 元数据过滤
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        # results 的结构：{"ids": [[...]], "documents": [[...]], ...}
        # 外层 list 只有 1 个元素（因为我们只传了 1 个 query）
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i] or {}
            distance = results["distances"][0][i]

            # Chroma 用 cosine 距离时，distance = 1 - cosine_similarity
            # 所以我们转换为 similarity，让分数越高表示越相关
            if self.collection.metadata.get("hnsw:space") == "cosine":
                score = 1.0 - distance
            else:
                score = -distance  # l2 距离：越小越好，取负数让排序一致

            # 阈值过滤
            if score_threshold is not None and score < score_threshold:
                continue

            retrieved.append(RetrievedChunk(
                content=text,
                metadata={**metadata, "doc_id": doc_id},
                score=score
            ))

        # 按分数从高到低排序（已经是这个顺序，但显式排序更保险）
        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        if ids is not None:
            self.collection.delete(ids=ids)
        elif filter_dict is not None:
            self.collection.delete(where=filter_dict)
        else:
            raise ValueError("必须提供 ids 或 filter_dict 之一")

    def count(self) -> int:
        return self.collection.count()

    def get_all(self, limit: int = 100000) -> List[RetrievedChunk]:
        """获取 collection 中的所有文档（用于构建 BM25 语料库）"""
        total = self.collection.count()
        if total == 0:
            return []
        
        results = self.collection.get(limit=min(limit, total))
        retrieved = []
        docs = results.get('documents', []) or []
        metas = results.get('metadatas', []) or []
        
        for i in range(len(docs)):
            retrieved.append(RetrievedChunk(
                content=docs[i],
                metadata=metas[i] if i < len(metas) else {},
                score=1.0
            ))
        return retrieved


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant 向量数据库（生产级推荐）
    
    特点：
    - 支持磁盘和内存混合存储，性能优秀
    - 丰富的过滤查询能力
    - 提供 Docker 镜像，部署简单
    
    启动 Qdrant（Docker）：
        docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    
    安装 Python 客户端：pip install qdrant-client
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: str = "Cosine"
    ):
        super().__init__(collection_name, dimension)
        
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("请安装 qdrant-client: pip install qdrant-client")

        # 默认连接本地 Qdrant
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.distance_metric = distance_metric

        # 确保 Collection 存在
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams

        # 检查 Collection 是否已存在
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # 创建新 Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance[self.distance_metric.upper()]
                )
            )
            print(f"[Qdrant] 创建 Collection: {self.collection_name}")

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        from qdrant_client.models import PointStruct

        if not texts:
            return []

        if ids is None:
            ids = self._generate_ids(len(texts))

        if metadatas is None:
            metadatas = [{} for _ in texts]

        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Qdrant 的 payload 中可以存任意 JSON 数据
            payload = {
                "content": text,
                **metadata
            }
            points.append(PointStruct(
                id=ids[i],
                vector=embedding,
                payload=payload
            ))

        # 批量插入（Qdrant 推荐每批 100~1000 条）
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # 构建 Qdrant 的过滤条件
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                from qdrant_client.models import Must
                qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )

        retrieved = []
        for point in results:
            retrieved.append(RetrievedChunk(
                content=point.payload.get("content", ""),
                metadata={k: v for k, v in point.payload.items() if k != "content"},
                score=point.score
            ))

        return retrieved

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        if ids is not None:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
        elif filter_dict is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            from qdrant_client.models import Must
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions)
            )
        else:
            raise ValueError("必须提供 ids 或 filter_dict 之一")

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count


# ═══════════════════════════════════════════════════════════
# 工厂类
# ═══════════════════════════════════════════════════════════

class VectorStoreFactory:
    """
    向量数据库工厂
    
    使用方式：
        # 开发环境
        store = VectorStoreFactory.create("chroma", collection_name="docs", dimension=1536)
        
        # 生产环境（只需改一行）
        store = VectorStoreFactory.create("qdrant", collection_name="docs", dimension=1536)
    """

    @staticmethod
    def create(
        provider: str,
        collection_name: str,
        dimension: int,
        **kwargs
    ) -> BaseVectorStore:
        """
        创建向量数据库实例
        
        Args:
            provider: "chroma" | "qdrant" | "milvus"
            collection_name: Collection 名称
            dimension: 向量维度（必须与 Embedding 模型一致）
            **kwargs: 各数据库特定的参数
        """
        if provider == "chroma":
            return ChromaVectorStore(collection_name, dimension, **kwargs)
        elif provider == "qdrant":
            return QdrantVectorStore(collection_name, dimension, **kwargs)
        else:
            raise ValueError(f"不支持的向量数据库: {provider}")


# ── 第5课小结 ──
#
# 你学到了：
# 1. 向量数据库用 ANN 算法（如 HNSW）实现亚秒级的百万级向量检索
# 2. Collection = 表，Distance Metric = 相似度计算方式，Payload = 附加元数据
# 3. Chroma 零配置启动，适合本地开发；Qdrant 性能更强，适合生产
# 4. 元数据过滤是实现"只搜某个文件"或"只搜 PDF"的关键能力
#
# 思考题：
# - 为什么 distance=cosine 时，score = 1 - distance？
#   （答案：cosine 距离 = 1 - cosine 相似度。相似度范围是 [-1, 1]，
#   但归一化后的向量相似度范围是 [0, 1]，所以距离范围也是 [0, 1]。
#   我们习惯用"分数越高越相关"，所以转换回相似度。）
#
# 下节课：RAG 检索链 —— 如何把 Embedding、向量库、重排序组装成完整的检索流程？
