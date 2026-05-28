"""Load env.txt and expose typed constants."""
import os
import tempfile
from pathlib import Path

_ENV_PATH = Path(__file__).parent / "env.txt"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env(_ENV_PATH)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")

DATA_DIR: Path = Path(__file__).parent / "data"
REPO_CLONE_DIR: Path = Path(tempfile.gettempdir()) / "code_tool_repos"
