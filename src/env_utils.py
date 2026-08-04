import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> bool:
    """Load environment variables from the project .env file regardless of cwd."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [project_root / ".env", Path.cwd() / ".env"]

    loaded = False
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            loaded = True

    return loaded


def get_api_key(*names: str) -> str:
    """Return the first non-empty API key from a list of environment variable names."""
    for name in names:
        value = os.getenv(name)
        if value and str(value).strip():
            return str(value).strip()
    return ""