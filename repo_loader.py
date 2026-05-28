"""Load a repository from GitHub URL or local path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from config import REPO_CLONE_DIR


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner/repo from GitHub URL. Returns (owner, repo)."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    raise ValueError(f"Invalid GitHub URL: {url}")


def _is_github_url(source: str) -> bool:
    """Check if source is a GitHub URL."""
    return source.startswith("https://github.com/") or source.startswith("git@github.com:")


def load_repo(source: str, force_reclone: bool = False) -> Path:
    """Load a repository from GitHub URL or local path.

    Args:
        source: GitHub URL or local filesystem path
        force_reclone: if True, delete and re-clone GitHub repos

    Returns:
        Path to the repository root

    Raises:
        ValueError: if source is invalid or clone fails
    """
    if _is_github_url(source):
        return _load_github(source, force_reclone)
    else:
        return _load_local(source)


def _load_github(url: str, force_reclone: bool = False) -> Path:
    """Clone a GitHub repository and return the local path."""
    owner, repo = _parse_github_url(url)
    repo_name = repo.lower()
    target_dir = REPO_CLONE_DIR / repo_name

    if target_dir.exists() and not force_reclone:
        print(f"[repo_loader] using cached repo at {target_dir}", file=sys.stderr)
        return target_dir

    if target_dir.exists() and force_reclone:
        print(f"[repo_loader] removing cached repo at {target_dir}", file=sys.stderr)
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)

    REPO_CLONE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[repo_loader] cloning {url} to {target_dir}", file=sys.stderr)
    try:
        result = subprocess.run(
            ["git", "clone", url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr}")
    except FileNotFoundError:
        raise RuntimeError("git command not found. Please install Git.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("git clone timed out after 300 seconds")

    print(f"[repo_loader] successfully cloned to {target_dir}", file=sys.stderr)
    return target_dir


def _load_local(path: str) -> Path:
    """Load a local repository and return its path."""
    repo_path = Path(path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Local path does not exist: {path}")
    if not repo_path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    print(f"[repo_loader] using local repo at {repo_path}", file=sys.stderr)
    return repo_path
