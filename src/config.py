from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.75


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _provider_config(prefix: str, default_provider: str, default_model: str) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    api_key_names = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    base_url_names = {
        "custom": "CUSTOM_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "openrouter": "OPENROUTER_BASE_URL",
    }
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", default_model),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(api_key_names.get(provider, "")),
        base_url=os.getenv(base_url_names.get(provider, "")),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=_int_env("COMPACT_THRESHOLD_TOKENS", 1200),
        compact_keep_messages=_int_env("COMPACT_KEEP_MESSAGES", 6),
        model=_provider_config("LLM", "openai", "gpt-4o-mini"),
        judge_model=_provider_config("JUDGE", "openai", "gpt-4o-mini"),
        profile_confidence_threshold=float(
            os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.75")
        ),
    )
