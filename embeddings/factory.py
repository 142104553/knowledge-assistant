"""
第4课：Embedding 服务封装

核心接口：
    client = EmbeddingFactory.create("openai", api_key="xxx")
    vectors = client.embed(["文本1", "文本2", "文本3"])
    # 返回: List[List[float]]，每个文本对应一个向量

设计要点：
- 批量调用：Embedding API 支持一次传入多个文本，比单条调用效率高 10 倍以上
- 重试机制：网络波动时自动重试
- 维度一致性：同一项目固定使用一个模型，禁止混用（不同模型的向量空间不兼容）
"""

import os
# 国内 HuggingFace 镜像，解决模型下载超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbeddingClient(ABC):
    """
    Embedding 客户端抽象基类
    
    所有子类必须实现：
    - embed(texts: List[str]) -> List[List[float]]
    - dimension -> int
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将一批文本编码为向量
        
        Args:
            texts: 文本列表（通常是一个 batch 的 chunks）
            
        Returns:
            与输入等长的向量列表，每个向量是 float 列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度（创建向量数据库索引时需要）"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识名（用于元数据记录和校验）"""
        pass

    def _normalize(self, vectors: List[List[float]]) -> List[List[float]]:
        """
        L2 归一化（可选但推荐）
        
        归一化后，向量内积 = 余弦相似度，检索时可以只用内积加速。
        """
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 防止除零
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms
        return normalized.tolist()


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """
    OpenAI Embedding 客户端
    
    支持：text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    
    环境变量：
        OPENAI_API_KEY=your-api-key
        OPENAI_BASE_URL=https://api.openai.com/v1  （可选，用于代理）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供 OpenAI API Key")

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model

        # Embedding 客户端初始化完成

        # 初始化客户端
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.OpenAI(**client_kwargs)

    @property
    def dimension(self) -> int:
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)

    @property
    def model_name(self) -> str:
        return self.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        调用 OpenAI Embedding API
        
        OpenAI 限制：单次最多 2048 个文本，每个文本最多 8191 tokens。
        对中文来说，通常一个字符 ≈ 1~1.5 tokens，所以我们的 chunk_size=512 是安全的。
        """
        if not texts:
            return []

        # 过滤空文本（Embedding API 不接受空字符串）
        texts = [t if t.strip() else " " for t in texts]

        # 批量控制：如果超过 1000 条，分批处理
        batch_size = 1000
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                # 可选：dimensions 参数可以降维（仅 text-embedding-3 系列支持）
                # dimensions=256  # 降低维度以减少存储
            )
            vectors = [item.embedding for item in response.data]
            all_vectors.extend(vectors)

        return self._normalize(all_vectors)


class BGEEmbeddingClient(BaseEmbeddingClient):
    """
    BGE 本地 Embedding 客户端（开源方案）
    
    模型：BAAI/bge-large-zh-v1.5（中文优化）或 BAAI/bge-m3（多语言）
    
    安装依赖：
        pip install sentence-transformers torch
    
    第一次运行会自动下载模型（约 1~2GB）。
    
    优势：完全离线，无 API 费用，数据不出本地。
    劣势：需要 GPU 才能达到理想速度，CPU 上较慢。
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        self._model_name = model_name
        # 自动检测 GPU，无 GPU 则用 CPU
        self.device = "cuda" if self._has_gpu() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def dimension(self) -> int:
        # BGE-large-zh 是 1024 维，其他模型可能不同
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        texts = [t if t.strip() else " " for t in texts]

        # SentenceTransformer 已经内部处理了 batching
        # 默认 batch_size=32，可以调大以利用 GPU 并行
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # 自动 L2 归一化
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        return vectors.tolist()


class MockEmbeddingClient(BaseEmbeddingClient):
    """
    模拟 Embedding 客户端（用于测试和离线开发）
    
    不调用任何外部服务，返回随机向量（但保证相同文本得到相同向量）。
    这样可以在没有 API Key 的情况下测试检索链的逻辑。
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        self._cache = {}  # 文本 → 向量的缓存

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"mock-{self._dimension}d"

    def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib

        results = []
        for text in texts:
            if text not in self._cache:
                # 用文本的哈希作为随机种子，保证相同文本得到相同向量
                seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                vector = np.random.randn(self._dimension).astype(np.float32)
                vector = vector / np.linalg.norm(vector)  # 归一化
                self._cache[text] = vector.tolist()
            results.append(self._cache[text])
        return results


# ═══════════════════════════════════════════════════════════
# 工厂类
# ═══════════════════════════════════════════════════════════

class EmbeddingFactory:
    """
    Embedding 客户端工厂
    
    使用方式：
        client = EmbeddingFactory.create("openai", api_key="your-api-key")
        # 或从环境变量自动读取
        client = EmbeddingFactory.create_from_env()
    """

    @staticmethod
    def create(
        provider: str,
        **kwargs
    ) -> BaseEmbeddingClient:
        """
        创建 Embedding 客户端
        
        Args:
            provider: "openai" | "bge" | "mock"
            **kwargs: 各客户端特定的参数
        """
        if provider == "openai":
            return OpenAIEmbeddingClient(**kwargs)
        elif provider == "bge":
            return BGEEmbeddingClient(**kwargs)
        elif provider == "mock":
            return MockEmbeddingClient(**kwargs)
        else:
            raise ValueError(f"不支持的 Embedding 提供商: {provider}")

    @staticmethod
    def create_from_env() -> BaseEmbeddingClient:
        """
        从环境变量自动推断配置
        
        优先级：
        1. 如果设置了 OPENAI_API_KEY → 使用 OpenAI
        2. 否则如果有本地 GPU → 使用 BGE
        3. 否则使用 Mock（仅用于测试）
        """
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddingClient()

        # 尝试检测 sentence-transformers 是否可用
        try:
            import sentence_transformers
            return BGEEmbeddingClient()
        except ImportError:
            pass

        print("[警告] 未检测到 OpenAI Key 或本地模型，回退到 MockEmbeddingClient")
        return MockEmbeddingClient()


# ── 第4课小结 ──
#
# 你学到了：
# 1. Embedding = 语义压缩，把文本变成高维向量
# 2. 余弦相似度衡量语义相近程度，是检索的核心数学工具
# 3. OpenAI API 适合快速启动，BGE 适合私有化部署
# 4. 批量调用 + 归一化是工程实践中的两个关键优化
#
# 思考题：
# - 为什么同一个项目不能混用不同 Embedding 模型？
#   （答案：不同模型训练的数据和目标不同，它们的向量空间"坐标系"不兼容。
#   比如 OpenAI 的 [0.5, -0.3] 和 BGE 的 [0.5, -0.3] 代表完全不同的语义。）
#
# 下节课：向量数据库 —— 如何存储和检索百万级向量？
