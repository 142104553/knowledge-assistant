"""
第6课（Part 2）：重排序器（Reranker）

重排序是 RAG 的"精排"阶段：
- 召回阶段（混合检索）：速度快，从百万文档中粗筛出几百个候选
- 重排序阶段：速度慢但精度高，对候选逐一精细打分，保留最相关的 Top-N

核心接口：
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large")
    ranked = reranker.rerank(query, candidates, top_n=5)
"""

from abc import ABC, abstractmethod
from typing import List

from models.document import RetrievedChunk


class BaseReranker(ABC):
    """重排序器抽象基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5
    ) -> List[RetrievedChunk]:
        """
        对候选 chunk 重新排序

        Args:
            query: 用户原始查询
            candidates: 召回阶段得到的候选 chunk 列表
            top_n: 返回最相关的 N 个

        Returns:
            按相关性分数排序的 chunk 列表
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    基于 Cross-Encoder 的重排序器

    Cross-Encoder 的原理：
    - 把 [query, document] 拼接成一对输入模型
    - 模型直接输出这对文本的相关性分数
    - 相比向量检索（分别编码 query 和 doc），Cross-Encoder 能看到两者的交互，精度更高

    模型推荐：
    - BAAI/bge-reranker-large（中文优化，开源）
    - BAAI/bge-reranker-base（轻量版）

    安装依赖：pip install sentence-transformers

    注意：Cross-Encoder 是计算密集型操作，只对召回的 Top-K（如 20~50）做重排，
    不要对全库文档做重排。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        # 构建 Cross-Encoder 的输入对：[(query, doc1), (query, doc2), ...]
        pairs = [(query, c.content) for c in candidates]

        # 批量打分（batch_size 可调大以利用 GPU）
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)

        # 组装结果
        ranked = []
        for chunk, score in zip(candidates, scores):
            ranked.append(RetrievedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                score=float(score)  # Cross-Encoder 直接输出相关性分数
            ))

        # 按分数从高到低排序
        ranked.sort(key=lambda x: x.score, reverse=True)

        # 只保留 Top-N
        return ranked[:top_n]


class NoOpReranker(BaseReranker):
    """
    空重排序器（用于快速测试或资源受限场景）

    不做任何重排，直接返回原列表的前 top_n 个。
    """

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5
    ) -> List[RetrievedChunk]:
        return candidates[:top_n]


class ScoreThresholdFilter(BaseReranker):
    """
    分数阈值过滤器

    配合其他重排序器使用，过滤掉低质量结果。
    如果所有结果都低于阈值，返回空列表（触发"拒答"机制）。
    """

    def __init__(self, threshold: float = 0.5, base_reranker: BaseReranker = None):
        self.threshold = threshold
        self.base_reranker = base_reranker or NoOpReranker()

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int = 5
    ) -> List[RetrievedChunk]:
        ranked = self.base_reranker.rerank(query, candidates, top_n=top_n * 2)
        filtered = [c for c in ranked if c.score >= self.threshold]
        return filtered[:top_n]
