"""
第6课（Part 1）：混合检索器

结合 Dense（向量）检索和 Sparse（BM25/关键词）检索，
解决单一检索方式的盲区。

核心接口：
    retriever = HybridRetriever(vector_store, texts_for_bm25)
    results = retriever.retrieve(query, query_embedding, top_k=10)
"""

from typing import List, Optional, Dict, Any
import numpy as np

from models.document import RetrievedChunk
from vectorstore.factory import BaseVectorStore


class BM25Retriever:
    """
    BM25 关键词检索器（Sparse Retrieval）

    BM25 是信息检索领域的经典算法，基于词频和逆文档频率。
    优点：对精确匹配（如产品型号、法律条文编号）非常敏感。
    缺点：不理解语义，"退款"和"退货"被视为不同词。

    安装依赖：pip install rank-bm25
    """

    def __init__(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            texts: 所有 chunk 的文本内容（用于构建词频统计）
            metadatas: 对应的元数据
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("请安装 rank-bm25: pip install rank-bm25")

        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]

        # 简单的中文分词：按字符切分（对 BM25 来说足够用，也可用 jieba 更精准）
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """文本分词：中文字符级 + 英文单词级"""
        import re
        # 提取中文字符和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        执行 BM25 检索

        Returns:
            按 BM25 分数排序的 RetrievedChunk，score 做了归一化到 [0, 1]
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = max(scores) if max(scores) > 0 else 1.0

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(RetrievedChunk(
                content=self.texts[idx],
                metadata=self.metadatas[idx],
                score=float(scores[idx] / max_score)  # 归一化到 [0, 1]
            ))

        return results


class HybridRetriever:
    """
    混合检索器：Dense + Sparse

    召回阶段使用两种检索方式，结果合并去重后交给重排序器精排。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_weight: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        执行混合检索

        Args:
            query: 用户原始查询文本（用于 BM25）
            query_embedding: 查询的向量（用于 Dense 检索）
            top_k: 最终返回的结果数（合并去重后）
            dense_weight: Dense 检索结果的权重（0~1），Sparse 权重 = 1 - dense_weight
            filter_dict: 元数据过滤条件

        合并策略：RRF（Reciprocal Rank Fusion）
        RRF 公式：score = Σ 1 / (k + rank)
        优点：不需要统一两种检索的分数尺度，只利用排序位置
        """
        # 1. Dense 检索
        dense_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # 多取一些，给合并留空间
            filter_dict=filter_dict
        )

        # 2. Sparse 检索（如果有的话）
        sparse_results = []
        if self.bm25_retriever:
            sparse_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

        # 3. RRF 融合
        return self._rrf_fusion(dense_results, sparse_results, top_k, k=60)

    def _rrf_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        top_k: int,
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion

        对两种检索的结果列表，按排名位置计算融合分数：
            rrf_score = 1/(k + dense_rank) + 1/(k + sparse_rank)

        出现在两种检索结果中的文档会获得更高的融合分数。
        """
        from collections import defaultdict

        scores = defaultdict(float)
        chunks = {}

        # 记录 Dense 排名
        for rank, chunk in enumerate(dense_results):
            key = chunk.content  # 用内容作为去键（也可以用 chunk.metadata 里的 id）
            scores[key] += 1.0 / (k + rank + 1)
            chunks[key] = chunk

        # 记录 Sparse 排名
        for rank, chunk in enumerate(sparse_results):
            key = chunk.content
            scores[key] += 1.0 / (k + rank + 1)
            if key not in chunks:
                chunks[key] = chunk

        # 按 RRF 分数排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        result = []
        for content, score in sorted_items[:top_k]:
            chunk = chunks[content]
            result.append(RetrievedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                score=score  # 这里 score 是 RRF 融合分数
            ))

        return result
