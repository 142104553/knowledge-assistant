"""
Milvus 向量数据库实现（基于 pymilvus 3.0 MilvusClient）

支持：
- Dense 向量检索（与当前 BGE-small-zh-v1.5 兼容）
- 元数据过滤（doc_id, source_file 等 via dynamic fields）
- 文档增删查
- Collection 清空
- Milvus Lite（本地文件模式，无需 Docker）
- Milvus Standalone（Docker 部署，生产级）

预留：
- Sparse 向量字段（后续接入 BGE-M3 等模型后可启用原生 hybrid search）
"""

import os
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
    - Milvus Lite 支持本地文件模式（无需 Docker）
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
            from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus>=2.4.0")

        self.metric_type = metric_type
        self.host = host
        self.port = port

        # 连接参数
        client_kwargs = {}
        if uri:
            client_kwargs["uri"] = uri
        else:
            client_kwargs["uri"] = f"http://{host}:{port}"
        if token:
            client_kwargs["token"] = token

        self.client = MilvusClient(**client_kwargs)

        # 创建 collection（如果不存在）
        if not self.client.has_collection(collection_name):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                ],
                enable_dynamic_field=True,
            )
            self.client.create_collection(collection_name=collection_name, schema=schema)

    @staticmethod
    def _flatten_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将元数据扁平化为 Milvus 动态字段。
        只保留标量类型（str/int/float/bool），复杂类型转为 str。
        """
        if not metadata:
            return {}
        flat = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                flat[f"meta_{k}"] = v
            else:
                flat[f"meta_{k}"] = str(v)
        return flat

    @staticmethod
    def _build_filter_expr(filter_dict: Optional[Dict[str, Any]]) -> Optional[str]:
        """将 filter_dict 转换为 Milvus 表达式"""
        if not filter_dict:
            return None
        conditions = []
        for k, v in filter_dict.items():
            field = f"meta_{k}"
            if isinstance(v, str):
                conditions.append(f'{field} == "{v}"')
            else:
                conditions.append(f"{field} == {v}")
        return " and ".join(conditions) if conditions else None

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """批量添加文本到 Milvus"""
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        data = []
        for text_id, text, emb, meta in zip(ids, texts, embeddings, metadatas):
            item = {
                "id": text_id,
                "vector": emb,
                "content": text,
            }
            item.update(self._flatten_metadata(meta))
            data.append(item)

        self.client.insert(collection_name=self.collection_name, data=data)
        return ids

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """批量添加 DocumentChunk 到 Milvus（兼容接口）"""
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Dense 向量相似度检索"""
        expr = self._build_filter_expr(filter_dict)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            filter=expr,
            output_fields=["content"],
        )

        retrieved = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                retrieved.append(RetrievedChunk(
                    content=entity.get("content", ""),
                    metadata={},
                    score=float(hit.get("distance", 0)),
                ))
        return retrieved

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """删除文档"""
        if ids is not None:
            self.client.delete(collection_name=self.collection_name, ids=ids)
        elif filter_dict is not None:
            expr = self._build_filter_expr(filter_dict)
            if expr:
                self.client.delete(collection_name=self.collection_name, filter=expr)
        else:
            raise ValueError("必须提供 ids 或 filter_dict 之一")

    def clear(self) -> None:
        """清空整个 collection"""
        self.client.drop_collection(collection_name=self.collection_name)
        from pymilvus import FieldSchema, CollectionSchema, DataType
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            ],
            enable_dynamic_field=True,
        )
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

    def count(self) -> int:
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return stats.get("row_count", 0)

    def get_all(self, limit: int = 100000) -> List[RetrievedChunk]:
        """获取 collection 中的所有文档"""
        count = self.count()
        if count == 0:
            return []

        results = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["content"],
            limit=min(limit, max(count, 1)),
        )

        retrieved = []
        for r in results:
            retrieved.append(RetrievedChunk(
                content=r.get("content", ""),
                metadata={},
                score=0.0,
            ))
        return retrieved
