"""
第8课（Part 1）：应用核心配置

统一管理项目的所有配置项，支持从环境变量和 .env 文件加载。
使用 Pydantic Settings 进行类型校验和默认值管理。
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用配置类

    优先级（从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """

    # === LLM 配置 ===
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3

    # === Embedding 配置 ===
    embedding_provider: str = "openai"  # openai / bge / mock
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # === 向量数据库配置 ===
    vectorstore_provider: str = "chroma"  # chroma / qdrant
    vectorstore_collection: str = "documents"
    chroma_persist_dir: str = "./chroma_db"
    qdrant_url: Optional[str] = "http://localhost:6333"

    # === RAG 配置 ===
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 15  # 增大召回量，确保多文件覆盖
    score_threshold: float = 0.5
    max_context_tokens: int = 8000  # 增大上下文窗口，容纳更多文件内容

    # === Agent 配置 ===
    enable_agent: bool = True

    # === 应用配置 ===
    app_name: str = "Enterprise Knowledge Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 允许 .env 中有未定义的字段，不报错


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例（单例模式，带缓存）

    使用 lru_cache 确保配置只加载一次，提升性能。
    """
    return Settings()


# ── 第8课开始 ──
#
# 你学到了：
# 1. Pydantic Settings 统一管理配置，自动从环境变量和 .env 文件加载
# 2. lru_cache 实现配置单例，避免重复读取
# 3. 配置分层：LLM / Embedding / 向量库 / RAG / Agent / 应用
#
# 接下来：FastAPI RESTful API 服务
