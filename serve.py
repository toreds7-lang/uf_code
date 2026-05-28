#!/usr/bin/env python3
"""FastAPI server for browsing code analysis results."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import code_graph
import code_summary
import llm_client
import rag
from config import DATA_DIR


@dataclass
class RepoContext:
    repo_name: str
    data_dir: Path
    graph: dict[str, Any]
    rag_index: rag.RagIndex
    summary_cache: dict[str, dict]
    highlights_cache: dict[str, dict]


# Global context
_repo_context: RepoContext | None = None


def load_repo_context(repo_name: str) -> RepoContext:
    """Load all data for a repository."""
    data_dir = DATA_DIR / repo_name

    # Load graph
    graph = code_graph.load_graph(data_dir)

    # Load RAG index
    rag_index = rag.RagIndex(data_dir)

    # Load summary cache
    summary_cache = code_summary.load_cache(data_dir)

    # Load highlights
    highlights_path = data_dir / "highlights.json"
    highlights_cache = {}
    if highlights_path.exists():
        try:
            highlights_cache = json.loads(highlights_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[serve] warning: could not load highlights: {e}", file=sys.stderr)

    return RepoContext(
        repo_name=repo_name,
        data_dir=data_dir,
        graph=graph,
        rag_index=rag_index,
        summary_cache=summary_cache,
        highlights_cache=highlights_cache,
    )


app = FastAPI(title="Code Repository Analyzer")


@app.get("/")
async def serve_viewer():
    """Serve the interactive SPA."""
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        raise HTTPException(status_code=404, detail="viewer.html not found")
    return FileResponse(viewer_path, media_type="text/html")


@app.get("/api/repos")
async def list_repos():
    """List available repositories."""
    repos = []
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir() and (item / "graph.json").exists():
                repos.append(item.name)
    return {"repos": sorted(repos)}


@app.post("/api/load-repo/{repo_name}")
async def load_repo_endpoint(repo_name: str):
    """Load a repository for browsing."""
    global _repo_context
    try:
        _repo_context = load_repo_context(repo_name)
        return {"status": "loaded", "repo_name": repo_name}
    except Exception as e:
        print(f"[serve] error loading repo: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


def _require_context():
    """Ensure a repo is loaded."""
    if _repo_context is None:
        raise HTTPException(status_code=400, detail="No repository loaded. Use /api/load-repo/{repo_name}")


@app.get("/api/graph")
async def get_graph():
    """Return the full knowledge graph."""
    _require_context()
    return _repo_context.graph


@app.get("/api/summary")
async def get_summary():
    """Return the summary tree."""
    _require_context()
    summary_path = _repo_context.data_dir / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="summary.json not found")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.get("/api/node/{node_id}")
async def get_node(node_id: str):
    """Return details of a single node."""
    _require_context()
    node = next(
        (n for n in _repo_context.graph.get("nodes", []) if n["id"] == node_id),
        None,
    )
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    summary = _repo_context.summary_cache.get(node_id, {}).get("summary", "")
    return {
        **node,
        "summary": summary,
    }


@app.get("/api/highlights")
async def get_highlights():
    """Return highlights (key concepts by file)."""
    _require_context()
    return _repo_context.highlights_cache


@app.get("/api/report")
async def get_report():
    """Return the architecture report."""
    _require_context()
    report_path = _repo_context.data_dir / "report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report.md not found")
    return {"report": report_path.read_text(encoding="utf-8")}


def _summarize_stream(repo_context: RepoContext) -> Iterator[str]:
    """Stream summarization progress as NDJSON."""
    def on_progress(msg: str):
        yield json.dumps({"status": "progress", "message": msg}) + "\n"

    cache = code_summary.summarize_all(repo_context.graph, repo_context.data_dir, on_progress)
    code_summary.save_tree(repo_context.graph, cache, repo_context.data_dir)

    yield json.dumps({"status": "complete", "nodes_summarized": len(cache)}) + "\n"


@app.post("/api/summarize")
async def summarize_all_endpoint():
    """Stream summarization of all nodes."""
    _require_context()
    return StreamingResponse(
        _summarize_stream(_repo_context),
        media_type="application/x-ndjson",
    )


def _summarize_node_stream(node_id: str, repo_context: RepoContext) -> Iterator[str]:
    """Stream summarization of a single node and its ancestors."""
    def on_progress(msg: str):
        yield json.dumps({"status": "progress", "message": msg}) + "\n"

    cache = code_summary.summarize_node_and_ancestors(
        node_id,
        repo_context.graph,
        repo_context.data_dir,
        on_progress,
    )
    code_summary.save_tree(repo_context.graph, cache, repo_context.data_dir)

    yield json.dumps({"status": "complete", "node_id": node_id}) + "\n"


@app.post("/api/summarize/node/{node_id}")
async def summarize_node_endpoint(node_id: str):
    """Stream summarization of a single node and ancestors."""
    _require_context()
    return StreamingResponse(
        _summarize_node_stream(node_id, _repo_context),
        media_type="application/x-ndjson",
    )


def _chat_stream(question: str, repo_context: RepoContext) -> Iterator[str]:
    """Stream Q&A response."""
    # Retrieve relevant context from RAG
    context_chunks = repo_context.rag_index.topk(question, k=5)
    context_text = "\n---\n".join([
        f"[{c['node_type']}:{c['name']}] {c['text']}"
        for c in context_chunks
    ])

    # Build prompt
    prompt = f"""You are a code understanding assistant. Answer the question using the provided code context.

CODE CONTEXT:
{context_text}

QUESTION:
{question}

Provide a concise, accurate answer. Cite specific functions or classes when relevant."""

    system = "You are a helpful code documentation assistant."

    for token in llm_client.stream_messages([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]):
        yield token


@app.post("/api/chat")
async def chat_endpoint(question: str):
    """Stream Q&A response."""
    _require_context()
    return StreamingResponse(
        _chat_stream(question, _repo_context),
        media_type="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    # Try to load a repo if arg provided
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
        try:
            _repo_context = load_repo_context(repo_name)
            print(f"[serve] loaded repo: {repo_name}", file=sys.stderr)
        except Exception as e:
            print(f"[serve] warning: could not auto-load {repo_name}: {e}", file=sys.stderr)

    uvicorn.run(app, host="0.0.0.0", port=8000)
