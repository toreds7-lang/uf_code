"""Chunking + embedding index for RAG search over code summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import llm_client


_CHUNK_CHARS = 2000
_OVERLAP_CHARS = 200


def _chunk_summary(node_id: str, summary: str, node_type: str, name: str) -> list[dict[str, Any]]:
    """Split a node summary into chunks with metadata."""
    chunks: list[dict[str, Any]] = []
    summary = summary.strip()
    if not summary:
        return chunks
    i = 0
    while i < len(summary):
        seg = summary[i : i + _CHUNK_CHARS]
        chunks.append({
            "node_id": node_id,
            "node_type": node_type,
            "name": name,
            "text": seg
        })
        if i + _CHUNK_CHARS >= len(summary):
            break
        i += _CHUNK_CHARS - _OVERLAP_CHARS
    return chunks


def build_index(summary_cache: dict[str, dict], data_dir: Path) -> None:
    """Embed all summaries from cache and save vectors + metadata to data_dir."""
    all_chunks: list[dict[str, Any]] = []
    for node_id, cache_entry in summary_cache.items():
        summary = cache_entry.get("summary", "")
        node_type = cache_entry.get("node_type", "unknown")
        name = cache_entry.get("name", "")
        chunks = _chunk_summary(node_id, summary, node_type, name)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[rag] no chunks to index", file=sys.stderr)
        return

    print(f"[rag] embedding {len(all_chunks)} chunks", file=sys.stderr)
    vecs_parts: list[np.ndarray] = []
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = [c["text"] for c in all_chunks[i : i + BATCH]]
        vecs_parts.append(llm_client.embed(batch))
    vecs = np.vstack(vecs_parts).astype(np.float32)

    np.savez_compressed(data_dir / "rag.npz", vecs=vecs)
    (data_dir / "rag.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rag] wrote rag.npz ({vecs.shape}) + rag.json", file=sys.stderr)


class RagIndex:
    """In-memory RAG index loaded once per process."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.vecs: np.ndarray | None = None
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        vpath = self.data_dir / "rag.npz"
        cpath = self.data_dir / "rag.json"
        if not (vpath.exists() and cpath.exists()):
            print(f"[rag] no index at {self.data_dir}", file=sys.stderr)
            return
        self.vecs = np.load(vpath)["vecs"]
        self.chunks = json.loads(cpath.read_text(encoding="utf-8"))

    def topk(self, query: str, k: int = 5, node_type_filter: str | None = None) -> list[dict[str, Any]]:
        """Return top-k chunks by cosine similarity."""
        if self.vecs is None or not self.chunks:
            return []
        q = llm_client.embed([query])  # (1, D), already normalized
        sims = (self.vecs @ q[0]).astype(np.float32)
        if node_type_filter is not None:
            mask = np.array([c["node_type"] == node_type_filter for c in self.chunks])
            sims = np.where(mask, sims, -1.0)
        order = np.argsort(-sims)[:k]
        return [
            {**self.chunks[i], "score": float(sims[i])}
            for i in order
            if sims[i] > -1.0
        ]
