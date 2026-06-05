"""
Milvus 向量数据库实现

支持：
- Dense 向量检索（与当前 BGE-small-zh-v1.5 兼容）
- 元数据过滤（doc_id, source_file 等）
- 文档增删查
- Collection 清空

预留：
- Sparse 向量字段（后续接入 BGE-M3 等模型后可启用原生 hybrid search）
"""

import json
import uuid
from typing import List, Optional, Dict, Any

from models.document import DocumentChunk, RetrievedChunk
from vectorstore.factory import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus 向量数据库实现

    相比 Chroma 的优势：
    - 原生支持百万级规模
    - 支持 GPU 索引加速
    - 2.4+ 支持 Sparse 向量和 Hybrid Search
    - 分布式部署能力
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: str = "19530",
        uri: Optional[str] = None,
        token: str = "",
        metric_type: str = "COSINE",
    ):
        super().__init__(collection_name, dimension)

        try:
            from pymilvus import connections, FieldSchema, CollectionSchema, DataType
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus>=2.4.0")

        self.host = host
        self.port = port
        self.uri = uri
        self.token = token
        self.metric_type = metric_type

        # 建立连接
        conn_kwargs = {"alias": "default"}
        if uri:
            conn_kwargs["uri"] = uri
            if token:
                conn_kwargs["token"] = token
        else:
            conn_kwargs["host"] = host
            conn_kwargs["port"] = port

        connections.connect(**conn_kwargs)

        # 准备 schema
        self._fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            # 预留 sparse 向量字段（当前未启用，后续接入 BGE-M3 后可使用）
            # FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        self._schema = CollectionSchema(self._fields, description="Document chunks")

        # 获取或创建 collection
        from pymilvus import Collection, utility
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
        else:
            self.collection = Collection(name=collection_name, schema=self._schema)
            self._create_index()

        # 加载 collection 到内存
        self.collection.load()

    def _create_index(self) -> None:
        """为 embedding 字段创建索引"""
        from pymilvus import Collection
        index_params = {
            "metric_type": self.metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)

    @staticmethod
    def _build_expr(filter_dict: Optional[Dict[str, Any]]) -> Optional[str]:
        """将 filter_dict 转换为 Milvus 的 expr 表达式"""
        if not filter_dict:
            return None
        conditions = []
        for k, v in filter_dict.items():
            if isinstance(v, str):
                conditions.append(f'metadata["{k}"] == "{v}"')
            else:
                conditions.append(f'metadata["{k}"] == {v}')
        return " and ".join(conditions)

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """批量添加文档到 Milvus"""
        if not chunks or not embeddings:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        contents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.collection.insert([ids, contents, embeddings, metadatas])
        self.collection.flush()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Dense 向量相似度检索"""
        expr = self._build_expr(filter_dict)

        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": 10},
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "metadata"],
        )

        retrieved = []
        for hits in results:
            for hit in hits:
                retrieved.append(RetrievedChunk(
                    content=hit.entity.get("content"),
                    metadata=hit.entity.get("metadata") or {},
                    score=float(hit.distance),
                ))
        return retrieved

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """删除文档"""
        if ids is not None:
            ids_str = ",".join([f'"{i}"' for i in ids])
            self.collection.delete(expr=f'id in [{ids_str}]')
        elif filter_dict is not None:
            expr = self._build_expr(filter_dict)
            if expr:
                self.collection.delete(expr=expr)
        else:
            raise ValueError("必须提供 ids 或 filter_dict 之一")

    def clear(self) -> None:
        """清空整个 collection"""
        from pymilvus import utility
        utility.drop_collection(self.collection_name)
        # 重新创建
        from pymilvus import Collection
        self.collection = Collection(name=self.collection_name, schema=self._schema)
        self._create_index()
        self.collection.load()

    def count(self) -> int:
        self.collection.flush()
        return self.collection.num_entities

    def get_all(self, limit: int = 100000) -> List[RetrievedChunk]:
        """获取 collection 中的所有文档"""
        total = self.collection.num_entities
        if total == 0:
            return []

        results = self.collection.query(
            expr='id != ""',
            output_fields=["content", "metadata"],
            limit=min(limit, max(total, 1)),
        )

        retrieved = []
        for r in results:
            retrieved.append(RetrievedChunk(
                content=r.get("content", ""),
                metadata=r.get("metadata") or {},
                score=0.0,
            ))
        return retrieved
