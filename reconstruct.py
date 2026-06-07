"""Source-grounded reverse-engineering pass.

Produces a *reconstruction blueprint* for a target repository: per-function
implementation contracts, per-module specs, and a repo-level build plan, fused
with deterministic facts pulled straight from the knowledge graph (exact
signatures, call edges, dependencies).

The output (`reconstruction.md`) is meant to be fed to an agentic / vibe-coding
workflow to rebuild a behaviorally-equivalent implementation from scratch.

Mirrors the patterns in `code_summary.py`: bottom-up post-order traversal over
`contains` edges, incremental atomic disk cache (resumable), prompts reloaded
fresh from `prompts/<node_type>_reconstruct.system.txt`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import llm_client

# Source budgets (chars) fed to the LLM, to stay within context windows.
_MAX_FUNC_SRC = 8000
_MAX_FILE_SRC = 16000


def _load_prompt(name: str) -> str:
    """Load prompts/<name>.system.txt fresh (so edits take effect immediately)."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.system.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")


# ── cache ────────────────────────────────────────────────────────────────────

def load_cache(data_dir: Path) -> dict[str, dict]:
    """Load the reconstruction cache from disk."""
    cache_path = data_dir / "reconstruct_cache.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[reconstruct] warning: could not load cache: {e}", file=sys.stderr)
    return {}


def save_cache(cache: dict[str, dict], data_dir: Path) -> None:
    """Atomically save the reconstruction cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "reconstruct_cache.json"
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_source(
    source_root: Path | None,
    rel_path: str | None,
    start: int | None = None,
    end: int | None = None,
    max_chars: int = _MAX_FILE_SRC,
) -> str:
    """Read source text (optionally a line range) from the original checkout."""
    if not source_root or not rel_path:
        return ""
    full_path = Path(source_root) / rel_path
    if not full_path.exists():
        return ""
    try:
        text = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if start:
        lines = text.splitlines()
        s = max(0, int(start) - 1)
        e = int(end) if end else len(lines)
        text = "\n".join(lines[s:e])
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# ... [source truncated for length]"
    return text


def _children(graph: dict[str, Any], node_id: str) -> list[str]:
    """Direct children via `contains` edges, in edge order."""
    return [
        e.get("target")
        for e in graph.get("edges", [])
        if e.get("rel") == "contains" and e.get("source") == node_id
    ]


def _signature(meta: dict[str, Any]) -> str:
    """Render '(a: T, b) -> R' from function metadata."""
    args = meta.get("args", []) or []
    parts = [
        f"{a['name']}: {a['annotation']}" if a.get("annotation") else a.get("name", "?")
        for a in args
    ]
    ret = meta.get("returns")
    return f"({', '.join(parts)})" + (f" -> {ret}" if ret else "")


def _fmt_imports(imports: list[dict]) -> list[str]:
    """One backticked import line per ImportInfo dict."""
    out = []
    for i in imports or []:
        if i.get("is_from"):
            names = ", ".join(i.get("names", []))
            out.append(f"from {i.get('module','')} import {names}")
        else:
            out.append(f"import {i.get('module','')}")
    return out


# ── per-node LLM inputs ──────────────────────────────────────────────────────

def _build_func_input(node: dict[str, Any], source_root: Path | None) -> str:
    meta = node.get("metadata", {})
    src = _read_source(
        source_root, meta.get("file"), meta.get("lineno"), meta.get("end_lineno"), _MAX_FUNC_SRC
    )
    lines = [
        f"Function: {node['name']}{_signature(meta)}",
        f"File: {meta.get('file', '?')}:{meta.get('lineno', '?')}",
    ]
    if meta.get("class"):
        lines.append(f"Method of class: {meta['class']}")
    if meta.get("decorators"):
        lines.append(f"Decorators: {', '.join(meta['decorators'])}")
    if meta.get("docstring"):
        lines.append(f"Docstring: {meta['docstring']}")
    lines += ["", "Source code:", "```", src or "(source unavailable)", "```"]
    return "\n".join(lines)


def _child_specs_block(graph: dict[str, Any], node_id: str, cache: dict[str, dict], header: str) -> str:
    """Concatenate cached child reconstruction specs under a header."""
    out: list[str] = []
    for cid in _children(graph, node_id):
        entry = cache.get(cid)
        if not entry:
            continue
        out.append(f"### {entry.get('node_type', '?')}: {entry.get('name', cid)}")
        out.append(entry.get("spec", ""))
    if not out:
        return ""
    return header + "\n" + "\n".join(out)


def _build_class_input(node: dict[str, Any], graph: dict[str, Any], cache: dict[str, dict]) -> str:
    meta = node.get("metadata", {})
    lines = [f"Class: {node['name']}"]
    if meta.get("bases"):
        lines.append(f"Bases: {', '.join(meta['bases'])}")
    if meta.get("docstring"):
        lines.append(f"Docstring: {meta['docstring']}")
    block = _child_specs_block(graph, node["id"], cache, "\nReconstruction contracts of methods:")
    if block:
        lines += ["", block]
    return "\n".join(lines)


def _build_file_input(
    node: dict[str, Any], graph: dict[str, Any], cache: dict[str, dict], source_root: Path | None
) -> str:
    meta = node.get("metadata", {})
    lines = [
        f"Module: {meta.get('module', node['name'])}  (path: {meta.get('path', '?')})",
        f"Language: {meta.get('language', '?')}",
    ]
    if meta.get("docstring"):
        lines.append(f"Module docstring: {meta['docstring']}")
    imps = _fmt_imports(meta.get("imports", []))
    if imps:
        lines.append("Imports: " + "; ".join(imps))
    block = _child_specs_block(graph, node["id"], cache, "\nReconstruction contracts of components:")
    if block:
        lines += ["", block]
    src = _read_source(source_root, meta.get("path"))
    lines += ["", "Full module source:", "```", src or "(source unavailable)", "```"]
    return "\n".join(lines)


def _build_group_input(node: dict[str, Any], graph: dict[str, Any], cache: dict[str, dict]) -> str:
    """Input for folder/repo nodes (synthesized from child specs)."""
    lines = [f"{node['type'].capitalize()}: {node['name']}"]
    block = _child_specs_block(graph, node["id"], cache, "\nReconstruction specs of children:")
    if block:
        lines += ["", block]
    return "\n".join(lines)


def _build_input(
    node: dict[str, Any], graph: dict[str, Any], cache: dict[str, dict], source_root: Path | None
) -> str:
    ntype = node.get("type", "unknown")
    if ntype == "function":
        return _build_func_input(node, source_root)
    if ntype == "class":
        return _build_class_input(node, graph, cache)
    if ntype == "file":
        return _build_file_input(node, graph, cache, source_root)
    return _build_group_input(node, graph, cache)


# ── driver ───────────────────────────────────────────────────────────────────

def reconstruct_all(
    graph: dict[str, Any],
    source_root: Path | None,
    data_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Bottom-up source-grounded reconstruction of every node. Caches incrementally."""
    cache = load_cache(data_dir)

    node_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    node_children: dict[str, list[str]] = {nid: [] for nid in node_by_id}
    for edge in graph.get("edges", []):
        if edge.get("rel") == "contains" and edge.get("source") in node_children:
            node_children[edge["source"]].append(edge.get("target"))

    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        for child_id in node_children.get(node_id, []):
            visit(child_id)

        node = node_by_id.get(node_id)
        if not node or node_id in cache:
            return

        ntype = node.get("type", "unknown")
        system_prompt = _load_prompt(f"{ntype}_reconstruct")
        if not system_prompt:
            print(f"[reconstruct] warning: no prompt for {ntype}", file=sys.stderr)
            cache[node_id] = {
                "spec": f"[Error: no reconstruct prompt for {ntype}]",
                "name": node["name"],
                "node_type": ntype,
            }
            save_cache(cache, data_dir)
            return

        if on_progress:
            on_progress(f"Reconstructing {ntype}: {node['name']}")

        user_input = _build_input(node, graph, cache, source_root)
        tokens: list[str] = []
        try:
            for token in llm_client.stream_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]):
                tokens.append(token)
        except Exception as e:
            print(f"[reconstruct] error on {node_id}: {e}", file=sys.stderr)
            cache[node_id] = {
                "spec": f"[Error: {e}]",
                "name": node["name"],
                "node_type": ntype,
            }
            save_cache(cache, data_dir)
            return

        cache[node_id] = {
            "spec": "".join(tokens).strip(),
            "name": node["name"],
            "node_type": ntype,
        }
        save_cache(cache, data_dir)

    all_ids = set(node_by_id)
    has_parent = {
        e.get("target") for e in graph.get("edges", []) if e.get("rel") == "contains"
    }
    for root_id in all_ids - has_parent:
        visit(root_id)

    return cache


# ── blueprint assembly (deterministic facts fused with LLM specs) ────────────

def _fmt_def(node: dict[str, Any]) -> str:
    return f"def {node['name']}{_signature(node.get('metadata', {}))}"


def save_markdown(graph: dict[str, Any], cache: dict[str, dict], data_dir: Path) -> None:
    """Assemble reconstruction.md from cached specs + deterministic graph facts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    repo_name = graph.get("repo_name", "repository")
    repo_node = next((n for n in graph.get("nodes", []) if n.get("type") == "repo"), None)

    lines = [
        f"# Reconstruction Blueprint: {repo_name}",
        f"*Generated: {graph.get('created_at', '')}*",
        "",
        "> Source-grounded reverse-engineering spec. Feed this to an agentic / vibe-coding "
        "workflow to rebuild a behaviorally-equivalent implementation from scratch — from these "
        "descriptions, without the original code. Each component is described at two tiers: a "
        "**vibe level** (functional/behavioral, enough to rebuild on its own) and a **code-level "
        "contract** (precise details to match exactly). Signatures, dependencies, and call edges "
        "are extracted deterministically from the code graph.",
        "",
    ]

    if repo_node and repo_node["id"] in cache:
        lines += ["## 1. Repository blueprint", "", cache[repo_node["id"]].get("spec", ""), ""]

    lines += ["## 2. Modules", ""]
    file_nodes = [n for n in graph.get("nodes", []) if n.get("type") == "file"]
    file_nodes.sort(key=lambda n: n.get("metadata", {}).get("path", n["name"]))

    for fnode in file_nodes:
        meta = fnode.get("metadata", {})
        path = meta.get("path", fnode["name"])
        lines += [f"### `{path}`", ""]

        if fnode["id"] in cache:
            lines += [cache[fnode["id"]].get("spec", ""), ""]

        # Deterministic: external dependencies
        imp_lines = _fmt_imports(meta.get("imports", []))
        if imp_lines:
            lines += ["**Dependencies (imports):**", ""]
            lines += [f"- `{il}`" for il in imp_lines]
            lines.append("")

        # Deterministic: in-repo module dependencies (imports edges)
        dep_files = [
            nodes[e["target"]].get("metadata", {}).get("path", nodes[e["target"]]["name"])
            for e in edges
            if e.get("rel") == "imports" and e.get("source") == fnode["id"] and e.get("target") in nodes
        ]
        if dep_files:
            lines += ["**In-repo dependencies:** " + ", ".join(f"`{d}`" for d in sorted(set(dep_files))), ""]

        child_ids = _children(graph, fnode["id"])
        classes = [nodes[c] for c in child_ids if nodes.get(c, {}).get("type") == "class"]
        funcs = [nodes[c] for c in child_ids if nodes.get(c, {}).get("type") == "function"]

        # Deterministic: public surface
        if classes or funcs:
            lines += ["**Public surface:**", ""]
            for cls in classes:
                bases = cls.get("metadata", {}).get("bases", [])
                base_str = f"({', '.join(bases)})" if bases else ""
                lines.append(f"- class `{cls['name']}{base_str}`")
                for mid in _children(graph, cls["id"]):
                    m = nodes.get(mid)
                    if m and m.get("type") == "function":
                        lines.append(f"  - `{_fmt_def(m)}`")
            for f in funcs:
                lines.append(f"- `{_fmt_def(f)}`")
            lines.append("")

        # Per-function contracts (LLM spec + deterministic call edges)
        contract_lines: list[str] = []

        def emit(m: dict[str, Any]) -> None:
            m_meta = m.get("metadata", {})
            loc = f"{m_meta.get('file', '?')}:{m_meta.get('lineno', '?')}"
            contract_lines.append(f"##### `{_fmt_def(m)}` — {loc}")
            contract_lines.append("")
            if m["id"] in cache:
                contract_lines.append(cache[m["id"]].get("spec", ""))
            callees = [
                nodes[e["target"]]["name"]
                for e in edges
                if e.get("rel") == "calls" and e.get("source") == m["id"] and e.get("target") in nodes
            ]
            if callees:
                contract_lines.append("")
                contract_lines.append("_Calls (in-repo):_ " + ", ".join(f"`{c}`" for c in sorted(set(callees))))
            contract_lines.append("")

        for cls in classes:
            for mid in _children(graph, cls["id"]):
                m = nodes.get(mid)
                if m and m.get("type") == "function":
                    emit(m)
        for f in funcs:
            emit(f)

        if contract_lines:
            lines += ["**Function contracts:**", ""] + contract_lines

    report_path = data_dir / "reconstruction.md"
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(report_path)
    print(f"[reconstruct] wrote {report_path}", file=sys.stderr)
