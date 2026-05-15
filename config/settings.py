from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """
    Load simple KEY=VALUE pairs from a local .env file into os.environ.

    This intentionally avoids adding an extra dependency for the first version
    of the project. Existing environment variables are not overwritten.
    """
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppSettings:
    """
    Central configuration object for app- and provider-level settings.
    """

    app_name: str = "AgentShell"
    default_theme: str = "light"
    provider_name: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    local_knowledge_embedding_provider: str = "local"
    local_knowledge_embedding_model: str = "hashing-v1"
    local_knowledge_embedding_chunker_version: str = "keyword-window-v1"
    local_knowledge_embedding_max_texts_per_call: int = 32
    local_knowledge_embedding_max_chars_per_text: int = 8000
    local_knowledge_embedding_max_total_chars_per_call: int = 64000
    derived_dataset_max_memory_mb: int = 512

    @classmethod
    def from_env(cls) -> "AppSettings":
        """
        Build settings from environment variables after loading the local .env file.
        """
        load_env_file()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_provider = os.getenv("LOCAL_KNOWLEDGE_EMBEDDING_PROVIDER") or (
            "openai" if openai_api_key else "local"
        )
        default_embedding_model = "text-embedding-3-small" if embedding_provider == "openai" else "hashing-v1"

        return cls(
            app_name=os.getenv("APP_NAME", "AgentShell"),
            default_theme=os.getenv("DEFAULT_THEME", "light"),
            provider_name=os.getenv("PROVIDER_NAME", "openai"),
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            local_knowledge_embedding_provider=embedding_provider,
            local_knowledge_embedding_model=os.getenv(
                "LOCAL_KNOWLEDGE_EMBEDDING_MODEL",
                default_embedding_model,
            ),
            local_knowledge_embedding_chunker_version=os.getenv(
                "LOCAL_KNOWLEDGE_EMBEDDING_CHUNKER_VERSION",
                "keyword-window-v1",
            ),
            local_knowledge_embedding_max_texts_per_call=_int_from_env(
                "LOCAL_KNOWLEDGE_EMBEDDING_MAX_TEXTS_PER_CALL",
                32,
            ),
            local_knowledge_embedding_max_chars_per_text=_int_from_env(
                "LOCAL_KNOWLEDGE_EMBEDDING_MAX_CHARS_PER_TEXT",
                8000,
            ),
            local_knowledge_embedding_max_total_chars_per_call=_int_from_env(
                "LOCAL_KNOWLEDGE_EMBEDDING_MAX_TOTAL_CHARS_PER_CALL",
                64000,
            ),
            derived_dataset_max_memory_mb=_int_from_env(
                "AGENTSHELL_DERIVED_DATASET_MAX_MEMORY_MB",
                512,
            ),
        )


def get_settings() -> AppSettings:
    """
    Convenience helper to load and return application settings.
    """
    return AppSettings.from_env()


def _int_from_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
