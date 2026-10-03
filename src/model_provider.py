from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {
        "google": "gemini",
        "google-genai": "gemini",
        "google_generative_ai": "gemini",
        "claude": "anthropic",
        "anthorpic": "anthropic",
        "open-router": "openrouter",
        "local": "ollama",
    }
    normalized = value.strip().lower().replace("_", "-")
    return aliases.get(normalized, normalized)


def build_chat_model(config: ProviderConfig):
    """Create a LangChain chat model using lazy provider imports."""

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        kwargs = {**common, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            anthropic_api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {**common}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        kwargs = {**common, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)

    raise ValueError(
        f"Unsupported provider {config.provider!r}. "
        "Expected openai, custom, gemini, anthropic, ollama, or openrouter."
    )
