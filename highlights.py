"""Per-file keyword/concept extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import llm_client


def _load_prompt() -> str:
    """Load highlight prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "code_highlight.system.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")


def highlight_file(
    file_path: str,
    source_code: str,
    cache: dict[str, dict],
) -> dict[str, dict]:
    """Extract key concepts from a single file."""
    if file_path in cache:
        return cache

    prompt = _load_prompt()
    if not prompt:
        print("[highlights] warning: no prompt", file=sys.stderr)
        cache[file_path] = {"concepts": []}
        return cache

    # Call LLM
    tokens = []
    try:
        for token in llm_client.stream_messages([
            {"role": "system", "content": prompt},
            {"role": "user", "content": source_code},
        ]):
            tokens.append(token)
    except Exception as e:
        print(f"[highlights] error processing {file_path}: {e}", file=sys.stderr)
        cache[file_path] = {"concepts": []}
        return cache

    response = "".join(tokens).strip()

    # Parse JSON response
    try:
        concepts = json.loads(response)
        if not isinstance(concepts, list):
            concepts = []
    except json.JSONDecodeError:
        print(f"[highlights] warning: invalid JSON from LLM for {file_path}", file=sys.stderr)
        concepts = []

    cache[file_path] = {
        "concepts": concepts,
        "source_length": len(source_code),
    }

    return cache


def highlight_all(
    files: dict[str, str],  # {file_path: source_code}
    data_dir: Path,
) -> dict[str, dict]:
    """Extract key concepts from all files."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    cache_path = data_dir / "highlights.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[highlights] warning: could not load cache: {e}", file=sys.stderr)

    # Process each file
    for file_path, source_code in files.items():
        if file_path in cache:
            continue

        print(f"[highlights] processing {file_path}", file=sys.stderr)
        cache = highlight_file(file_path, source_code, cache)

        # Save incrementally
        tmp_path = cache_path.with_suffix(".tmp")
        json_text = json.dumps(cache, ensure_ascii=False, indent=2)
        tmp_path.write_text(json_text, encoding="utf-8")
        tmp_path.replace(cache_path)

    print(f"[highlights] wrote {cache_path}", file=sys.stderr)
    return cache
