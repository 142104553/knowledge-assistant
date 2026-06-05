"""
LLM 工厂

支持多模型路由，根据配置创建对应 Provider 的 LLM 客户端实例。
当前支持：MiMo (OpenAI-compatible) / Kimi (OpenAI-compatible) / OpenAI
"""

from typing import Optional, Union


def create_llm_client(client_type: str, provider: str, settings):
    """
    根据 provider 配置创建 LLM 客户端实例

    Args:
        client_type: "openai" 或 "langchain"
        provider: "mimo" / "kimi" / "openai"
        settings: app.core.config.Settings 配置对象

    Returns:
        LLMClient 或 LangChainLLMClient 实例
    """
    provider = (provider or "openai").lower()

    # 根据 provider 选择对应的 api_key / base_url / model
    if provider == "kimi":
        api_key = settings.kimi_api_key
        base_url = settings.kimi_base_url
        model = settings.kimi_model
    elif provider == "mimo":
        # MiMo 使用 OpenAI-compatible 接口，复用 openai_api_key 等字段
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        model = settings.llm_model or "MiMo-v2.5"
    else:
        # 默认 openai
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        model = settings.llm_model

    if not api_key:
        raise ValueError(f"Provider '{provider}' 缺少 API Key，请在 .env 中配置")

    client_type = (client_type or "openai").lower()

    if client_type == "langchain":
        from rag.chains.rag_chain import LangChainLLMClient
        return LangChainLLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=settings.llm_temperature
        )
    else:
        from rag.chains.rag_chain import LLMClient
        return LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model
        )


def get_provider_info(settings) -> dict:
    """返回当前默认 provider 的信息摘要"""
    provider = (settings.llm_default_provider or "openai").lower()
    if provider == "kimi":
        return {"provider": provider, "model": settings.kimi_model, "base_url": settings.kimi_base_url}
    elif provider == "mimo":
        return {"provider": provider, "model": settings.llm_model, "base_url": settings.openai_base_url}
    else:
        return {"provider": provider, "model": settings.llm_model, "base_url": settings.openai_base_url}
