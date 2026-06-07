# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A read-only tool that analyzes a Python (optionally JS/TS) repository and produces: a knowledge graph (nodes: repo/folder/file/class/function; edges: contains/imports/calls/inherits), bottom-up LLM summaries, per-file highlights, a RAG index over the summaries, and a Markdown report. A FastAPI server + single-file SPA (`viewer.html`) browse the results and answer questions via RAG.

There is **no test suite, linter, or build step** configured. The project is a set of plain Python scripts run directly. `requirements.txt` lists deps; a `.venv/` is already present.

## Commands

Use the venv interpreter directly on Windows (`.venv\Scripts\python.exe`) or activate with `.venv\Scripts\Activate.ps1`.

```powershell
# Install deps (into the venv)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Full analysis pipeline (GitHub URL or local path) -> writes data/<repo_name>/
.venv\Scripts\python.exe run.py https://github.com/owner/repo
.venv\Scripts\python.exe run.py D:\path\to\repo

# Parse only — builds graph.json, NO LLM calls, no cost (fast smoke test)
.venv\Scripts\python.exe run.py <source> --steps parse

# Stop after a given stage: parse | summarize | highlights | rag | (default) all
.venv\Scripts\python.exe run.py <source> --steps summarize
.venv\Scripts\python.exe run.py <source> --force   # bypass clone/cache, re-run

# Serve the viewer (auto-loads the named repo's data/<name>/) at http://localhost:8000
.venv\Scripts\python.exe serve.py <repo_name>
```

Note: `run.py --filter` is declared but currently a **no-op** (not wired up). Re-running `run.py` is the normal way to resume — every expensive step is cached on disk and skipped if already done.

## Architecture & data flow

The pipeline is a linear sequence orchestrated by [run.py](run.py); the server in [serve.py](serve.py) reads the same `data/<repo>/` artifacts and adds on-demand LLM endpoints. Stages:

1. [repo_loader.py](repo_loader.py) — local path, or `git clone` GitHub URLs into the OS temp dir (`%TEMP%/code_tool_repos/<repo>`, cloned dir name is lowercased).
2. [code_parser.py](code_parser.py) — `parse_repo()` walks files (skips `.venv`, `node_modules`, etc.), extracts classes/functions/imports/call-sites via stdlib `ast`. `resolve_call_graph()` does **three-tier best-effort** call resolution (qualified match → same-file name match → `self.method` in class); unresolved/stdlib calls are dropped.
3. [code_graph.py](code_graph.py) — `build_graph()` turns parsed data + call graph into the `{nodes, edges}` dict and writes `graph.json`. Node IDs are stable and hierarchical: `"<type>::<path>::<class>::<func>"` (`::` separator, spaces/slashes sanitized to `_`). **All traversal elsewhere depends on this ID scheme and on `rel == "contains"` edges.**
4. [code_summary.py](code_summary.py) — `summarize_all()` does a **post-order** DFS over `contains` edges: leaves (functions) summarized first, then classes/files/folders/repo synthesized from child summaries. Each node is cached to `summary_cache.json` immediately after generation (atomic `.tmp` → `replace`), so an interrupted run resumes. `save_tree()` emits the nested `summary.json`.
5. [highlights.py](highlights.py) — per-file LLM extraction of key identifiers; the prompt **must return a JSON array of strings** (parsed with `json.loads`). Cached in `highlights.json`.
6. [rag.py](rag.py) — chunks summaries (2000 chars, 200 overlap), embeds in batches, L2-normalizes, writes `rag.npz` (vectors) + `rag.json` (chunk metadata). `RagIndex.topk()` does cosine search (dot product of normalized vectors).
7. [report.py](report.py) — renders `report.md` from the summary tree (functions are intentionally omitted for brevity).

[viewer.html](viewer.html) is a single-file SPA (vis-network graph + tree + chat); served verbatim from `/`.

## Conventions & gotchas

- **LLM backend is OpenAI, not Claude.** [llm_client.py](llm_client.py) uses LangChain `init_chat_model(..., model_provider="openai")` with default model `gpt-4o-mini`. (The README/IMPLEMENTATION_SUMMARY prose saying responses come "from Claude" is stale.) Point `LLM_BASE_URL`/`EMBEDDING_BASE_URL` at a vLLM/OpenAI-compatible endpoint to swap providers (then `api_key` is sent as `"EMPTY"`).
- **Config comes from `env.txt`** (NOT `.env`), loaded by [config.py](config.py) via `os.environ.setdefault` (real env vars win). Keys: `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `LLM_BASE_URL`, `EMBEDDING_BASE_URL`. `env.txt` is gitignored; `env_example copy.txt` is the template. `DATA_DIR` in config.py is hardcoded to `./data` (the `DATA_DIR` env var is not actually read).
- **Prompt files drive all LLM behavior** and are reloaded fresh on every call — edit `prompts/*.txt` to change output, no code change needed. Summarization loads `prompts/<node_type>_describe.system.txt`, so the names must match node types exactly: `function_describe`, `class_describe`, `file_describe`, `folder_describe`, `repo_describe` (the docs' `func_describe` name is wrong). `func_explain.system.txt` is separate — used only by the server's `/api/explain/*` endpoints, not by `run.py`.
- **Caching is per-artifact and resumable.** Deleting a specific JSON in `data/<repo>/` forces just that stage to regenerate. `summary_cache.json` is the flat cache; `summary.json` is the derived tree; `explain_cache.json` and `source_root.txt` are written by the server, not `run.py`.
- **Server endpoints (in [serve.py](serve.py)) beyond what the README lists:** chat is `GET /api/chat?question=...` (not POST), plus `GET /api/source/{id}`, `GET /api/explain/node/{id}` and `/api/explain/class/{id}` (live LLM, cached in `explain_cache.json`), and `GET /api/mermaid`. The server holds a single global `_repo_context`; one repo is loaded at a time via `POST /api/load-repo/{name}`. `/api/source` needs `source_root.txt` to point at an on-disk checkout, so source/explain only work for repos analyzed locally on this machine.
- **JS/TS parsing is effectively disabled** — it requires the optional `py_tree_sitter_languages` import, which is absent, so only Python is parsed in practice. The tree-sitter `_parse_*` methods in code_parser.py are dead code unless that dep is installed.
