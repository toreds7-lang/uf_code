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
import reconstruct
from config import DATA_DIR, REPO_CLONE_DIR


@dataclass
class RepoContext:
    repo_name: str
    data_dir: Path
    source_root: Path | None
    graph: dict[str, Any]
    rag_index: rag.RagIndex
    summary_cache: dict[str, dict]
    highlights_cache: dict[str, dict]
    explain_cache: dict[str, str]
    reconstruct_cache: dict[str, dict]


def _load_explain_cache(data_dir: Path) -> dict[str, str]:
    """Load the per-node explanation cache from disk."""
    cache_path = data_dir / "explain_cache.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[serve] warning: could not load explain cache: {e}", file=sys.stderr)
    return {}


def _save_explain_cache(cache: dict[str, str], data_dir: Path) -> None:
    """Atomically save the explanation cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "explain_cache.json"
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)


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

    # Resolve original source root for the /api/source endpoint
    source_root: Path | None = None
    source_root_file = data_dir / "source_root.txt"
    if source_root_file.exists():
        candidate = Path(source_root_file.read_text(encoding="utf-8").strip())
        if candidate.is_dir():
            source_root = candidate
    if source_root is None:
        candidate = REPO_CLONE_DIR / repo_name
        if candidate.is_dir():
            source_root = candidate

    return RepoContext(
        repo_name=repo_name,
        data_dir=data_dir,
        source_root=source_root,
        graph=graph,
        rag_index=rag_index,
        summary_cache=summary_cache,
        highlights_cache=highlights_cache,
        explain_cache=_load_explain_cache(data_dir),
        reconstruct_cache=reconstruct.load_cache(data_dir),
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


@app.get("/api/source/{node_id:path}")
async def get_source(node_id: str):
    """Return raw source code for a node."""
    _require_context()
    node = next(
        (n for n in _repo_context.graph.get("nodes", []) if n["id"] == node_id),
        None,
    )
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    metadata = node.get("metadata", {})
    rel_path = metadata.get("file") or metadata.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="No file path for this node")

    if _repo_context.source_root is None:
        raise HTTPException(status_code=503, detail="Source root not available")

    full_path = (_repo_context.source_root / rel_path).resolve()
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found on disk")

    source = full_path.read_text(encoding="utf-8", errors="replace")
    lineno = metadata.get("lineno")
    return {"source": source, "lineno": lineno, "file": rel_path}


def _extract_source_snippet(node: dict, source_root: Path | None, max_lines: int = 80) -> str:
    """Read up to max_lines of source starting at the node's definition line."""
    if source_root is None:
        return ""
    meta = node.get("metadata", {})
    rel_path = meta.get("file") or meta.get("path")
    if not rel_path:
        return ""
    full_path = (source_root / rel_path).resolve()
    if not full_path.exists():
        return ""
    lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    lineno = meta.get("lineno")
    start = max(0, int(lineno) - 1) if lineno else 0
    return "\n".join(lines[start:start + max_lines])


def _explain_node_stream(node: dict, source_snippet: str) -> Iterator[str]:
    """Stream a code-level explanation for a function/method node."""
    prompt_path = Path(__file__).parent / "prompts" / "func_explain.system.txt"
    system = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else (
        "Explain how this function works in detail using markdown and bullet points."
    )

    meta = node.get("metadata", {})
    args = meta.get("args", [])
    arg_strs = [f"{a['name']}: {a.get('annotation', 'unknown')}" for a in args]
    user = "\n".join(filter(None, [
        f"Function: {node['name']}",
        f"Signature: def {node['name']}({', '.join(arg_strs)}) -> {meta.get('returns', 'unknown')}",
        f"Docstring: {meta['docstring']}" if meta.get("docstring") else None,
        f"Decorators: {', '.join(meta['decorators'])}" if meta.get("decorators") else None,
        "",
        "Source code:",
        "```python",
        source_snippet,
        "```",
    ]))

    for token in llm_client.stream_messages([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]):
        yield token


def _explain_node_cached(node: dict, repo_context: RepoContext) -> Iterator[str]:
    """Yield an explanation for a node, serving from disk cache when available.

    On a cache miss, streams from the LLM, accumulates the full text, then
    persists it to explain_cache.json so future requests are served locally.
    """
    node_id = node["id"]
    cached = repo_context.explain_cache.get(node_id)
    if cached:
        yield cached
        return

    source_snippet = _extract_source_snippet(node, repo_context.source_root)
    chunks: list[str] = []
    for token in _explain_node_stream(node, source_snippet):
        chunks.append(token)
        yield token

    # Persist the freshly generated explanation
    full_text = "".join(chunks).strip()
    if full_text:
        repo_context.explain_cache[node_id] = full_text
        try:
            _save_explain_cache(repo_context.explain_cache, repo_context.data_dir)
        except Exception as e:
            print(f"[serve] warning: could not save explain cache: {e}", file=sys.stderr)


@app.get("/api/explain/node/{node_id:path}")
async def explain_node_endpoint(node_id: str, refresh: bool = False):
    """Stream a code-level explanation for a function or method node.

    Served from explain_cache.json when present; pass ?refresh=true to regenerate.
    """
    _require_context()
    node = next(
        (n for n in _repo_context.graph.get("nodes", []) if n["id"] == node_id),
        None,
    )
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    if node.get("type") not in ("function", "method"):
        raise HTTPException(status_code=400, detail="Explain is only supported for function/method nodes")

    if refresh:
        _repo_context.explain_cache.pop(node_id, None)

    return StreamingResponse(
        _explain_node_cached(node, _repo_context),
        media_type="text/plain; charset=utf-8",
    )


def _explain_class_stream(class_node: dict, method_nodes: list[dict], repo_context: RepoContext) -> Iterator[str]:
    """Stream NDJSON lines — one per method — with full markdown explanations.

    Each method is served from / written to the shared explain cache, so methods
    explained individually (or in a prior run) are not regenerated.
    """
    for method in method_nodes:
        was_cached = method["id"] in repo_context.explain_cache
        markdown = "".join(_explain_node_cached(method, repo_context))
        yield json.dumps({
            "node_id": method["id"],
            "name": method["name"],
            "cached": was_cached,
            "markdown": markdown,
        }) + "\n"
    yield json.dumps({"status": "complete", "class": class_node["name"]}) + "\n"


@app.get("/api/explain/class/{class_node_id:path}")
async def explain_class_endpoint(class_node_id: str):
    """Stream NDJSON explanations for every function/method in a class."""
    _require_context()
    class_node = next(
        (n for n in _repo_context.graph.get("nodes", []) if n["id"] == class_node_id),
        None,
    )
    if not class_node:
        raise HTTPException(status_code=404, detail=f"Node not found: {class_node_id}")
    if class_node.get("type") != "class":
        raise HTTPException(status_code=400, detail="Endpoint only supports class nodes")

    # Collect direct function/method children via 'contains' edges
    child_ids = {
        e["target"]
        for e in _repo_context.graph.get("edges", [])
        if e.get("rel") == "contains" and e.get("source") == class_node_id
    }
    method_nodes = [
        n for n in _repo_context.graph.get("nodes", [])
        if n["id"] in child_ids and n.get("type") in ("function", "method")
    ]
    if not method_nodes:
        raise HTTPException(status_code=404, detail="No function/method children found for this class")

    return StreamingResponse(
        _explain_class_stream(class_node, method_nodes, _repo_context),
        media_type="application/x-ndjson",
    )


@app.get("/api/report")
async def get_report():
    """Return the architecture report."""
    _require_context()
    report_path = _repo_context.data_dir / "report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report.md not found")
    return {"report": report_path.read_text(encoding="utf-8")}


@app.get("/api/reconstruction")
async def get_reconstruction():
    """Return the reverse-engineering reconstruction blueprint (reconstruction.md)."""
    _require_context()
    recon_path = _repo_context.data_dir / "reconstruction.md"
    if not recon_path.exists():
        raise HTTPException(status_code=404, detail="reconstruction.md not found — run reconstruct first")
    return {"reconstruction": recon_path.read_text(encoding="utf-8")}


@app.get("/api/reconstruct/node/{node_id:path}")
async def get_reconstruct_node(node_id: str):
    """Return the cached reconstruction contract for a single node."""
    _require_context()
    entry = _repo_context.reconstruct_cache.get(node_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No reconstruction spec for node: {node_id}")
    return entry


def _reconstruct_stream(repo_context: RepoContext) -> Iterator[str]:
    """Run the reconstruction pass and stream a completion line (NDJSON)."""
    if repo_context.source_root is None:
        yield json.dumps({
            "status": "error",
            "message": "Original source not available on this host; run `run.py <repo> --steps reconstruct` locally.",
        }) + "\n"
        return

    cache = reconstruct.reconstruct_all(
        repo_context.graph, repo_context.source_root, repo_context.data_dir
    )
    reconstruct.save_markdown(repo_context.graph, cache, repo_context.data_dir)
    repo_context.reconstruct_cache = cache

    yield json.dumps({"status": "complete", "nodes_reconstructed": len(cache)}) + "\n"


@app.post("/api/reconstruct")
async def reconstruct_endpoint():
    """Generate the reverse-engineering reconstruction blueprint for the loaded repo."""
    _require_context()
    return StreamingResponse(
        _reconstruct_stream(_repo_context),
        media_type="application/x-ndjson",
    )


def _generate_mermaid(graph: dict[str, Any]) -> str:
    """Generate a Mermaid flowchart from file-level import edges."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    file_nodes = {n["id"]: n for n in nodes if n["type"] == "file"}
    if not file_nodes:
        return "graph TD\n  A[No files found]"

    # Build safe ID mapping (mermaid can't handle :: or /)
    id_map: dict[str, str] = {nid: f"F{i}" for i, nid in enumerate(file_nodes)}

    import_edges = [
        e for e in edges
        if e["rel"] == "imports" and e["source"] in id_map and e["target"] in id_map
    ]

    # Group files by top-level folder
    from pathlib import Path as _Path
    folder_map: dict[str, list[str]] = {}
    for nid, node in file_nodes.items():
        file_path = node.get("metadata", {}).get("path", node["name"])
        parts = _Path(file_path).parts
        top_folder = parts[0] if len(parts) > 1 else ""
        folder_map.setdefault(top_folder, []).append(nid)

    lines = ["graph TD"]

    for folder, file_ids in sorted(folder_map.items()):
        if folder:
            safe = folder.replace("-", "_").replace(".", "_")
            lines.append(f'  subgraph {safe}["{folder}"]')
            for fid in file_ids:
                label = file_nodes[fid]["name"]
                lines.append(f'    {id_map[fid]}["{label}"]')
            lines.append("  end")
        else:
            for fid in file_ids:
                label = file_nodes[fid]["name"]
                lines.append(f'  {id_map[fid]}["{label}"]')

    for edge in import_edges:
        lines.append(f'  {id_map[edge["source"]]} --> {id_map[edge["target"]]}')

    return "\n".join(lines)


@app.get("/api/mermaid")
async def get_mermaid():
    """Return a Mermaid flowchart of the file-level import graph."""
    _require_context()
    return {"mermaid": _generate_mermaid(_repo_context.graph)}


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


@app.get("/api/chat")
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
