from __future__ import annotations


APP_ID = "local_knowledge"
APP_LABEL = "Local Knowledge"
APP_TYPE = "streamlit"

LOCAL_KNOWLEDGE_STATE_KEY = "local_knowledge_state"

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".docx",
    ".json",
    ".md",
    ".pdf",
    ".pptx",
    ".py",
    ".txt",
    ".xls",
    ".xlsm",
    ".xlsx",
}

DATASET_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsm",
    ".xlsx",
}

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

IGNORED_FILE_NAMES = {
    ".DS_Store",
}
