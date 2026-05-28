"""Bottom-up hierarchical LLM summarization of code nodes."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import llm_client


def _load_prompt(name: str) -> str:
    """Load prompt template from prompts/ directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.system.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")


def _build_func_input(node: dict[str, Any], graph: dict[str, Any]) -> str:
    """Format function metadata for LLM summarization."""
    meta = node.get("metadata", {})
    lines = [
        f"Function: {node['name']}",
        f"File: {meta.get('file', 'unknown')}",
    ]

    args = meta.get("args", [])
    if args:
        arg_strs = [f"{a['name']} ({a.get('annotation', 'unknown')})" for a in args]
        lines.append(f"Inputs: {', '.join(arg_strs)}")
    else:
        lines.append("Inputs: (none)")

    returns = meta.get("returns", "unknown")
    lines.append(f"Outputs: {returns}")

    if meta.get("docstring"):
        lines.append(f"Docstring: {meta['docstring']}")

    if meta.get("decorators"):
        lines.append(f"Decorators: {', '.join(meta['decorators'])}")

    return "\n".join(lines)


def _build_parent_input(node: dict[str, Any], graph: dict[str, Any], cache: dict[str, dict]) -> str:
    """Format parent node with child summaries for LLM."""
    lines = [f"{node['type'].capitalize()}: {node['name']}"]

    # Find children via contains edges
    children_ids = []
    for edge in graph.get("edges", []):
        if edge.get("rel") == "contains" and edge.get("source") == node["id"]:
            children_ids.append(edge.get("target"))

    if not children_ids:
        return "\n".join(lines)

    lines.append("\nChild summaries:")
    for child_id in children_ids:
        if child_id in cache:
            summary = cache[child_id].get("summary", "").split("\n")[0]
            child_name = cache[child_id].get("name", child_id)
            lines.append(f"  - {child_name}: {summary[:80]}...")

    return "\n".join(lines)


def load_cache(data_dir: Path) -> dict[str, dict]:
    """Load the summary cache from disk."""
    cache_path = data_dir / "summary_cache.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[summary] warning: could not load cache: {e}", file=sys.stderr)
    return {}


def save_cache(cache: dict[str, dict], data_dir: Path) -> None:
    """Atomically save the summary cache."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "summary_cache.json"
    tmp_path = cache_path.with_suffix(".tmp")

    json_text = json.dumps(cache, ensure_ascii=False, indent=2)
    tmp_path.write_text(json_text, encoding="utf-8")
    tmp_path.replace(cache_path)


def summarize_all(
    graph: dict[str, Any],
    data_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Bottom-up summarization of all nodes. Caches incrementally."""
    cache = load_cache(data_dir)

    # Build contains-tree for traversal
    node_children: dict[str, list[str]] = {}
    for node in graph.get("nodes", []):
        node_children[node["id"]] = []

    for edge in graph.get("edges", []):
        if edge.get("rel") == "contains":
            source = edge.get("source")
            target = edge.get("target")
            if source in node_children:
                node_children[source].append(target)

    # Post-order traversal
    visited = set()

    def traverse(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)

        # Visit children first
        for child_id in node_children.get(node_id, []):
            traverse(child_id)

        # Summarize this node
        node = next((n for n in graph.get("nodes", []) if n["id"] == node_id), None)
        if not node:
            return

        if node_id in cache:
            return  # Already summarized

        node_type = node.get("type", "unknown")

        if node_type == "function":
            prompt_input = _build_func_input(node, graph)
        else:
            prompt_input = _build_parent_input(node, graph, cache)

        if on_progress:
            on_progress(f"Summarizing {node_type}: {node['name']}")

        # Call LLM
        system_prompt = _load_prompt(f"{node_type}_describe")
        if not system_prompt:
            print(f"[summary] warning: no prompt for {node_type}", file=sys.stderr)
            cache[node_id] = {
                "summary": f"[Error: no prompt for {node_type}]",
                "name": node["name"],
                "node_type": node_type,
            }
            save_cache(cache, data_dir)
            return

        summary_tokens = []
        try:
            for token in llm_client.stream_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_input},
            ]):
                summary_tokens.append(token)
        except Exception as e:
            print(f"[summary] error summarizing {node_id}: {e}", file=sys.stderr)
            cache[node_id] = {
                "summary": f"[Error: {e}]",
                "name": node["name"],
                "node_type": node_type,
            }
            save_cache(cache, data_dir)
            return

        summary = "".join(summary_tokens).strip()
        cache[node_id] = {
            "summary": summary,
            "name": node["name"],
            "node_type": node_type,
        }
        save_cache(cache, data_dir)

    # Find all root nodes (no incoming contains edges)
    all_node_ids = {n["id"] for n in graph.get("nodes", [])}
    has_parent = set()
    for edge in graph.get("edges", []):
        if edge.get("rel") == "contains":
            has_parent.add(edge.get("target"))

    roots = all_node_ids - has_parent

    for root_id in roots:
        traverse(root_id)

    return cache


def summarize_node_and_ancestors(
    node_id: str,
    graph: dict[str, Any],
    data_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Summarize a single node and its ancestors (for partial updates)."""
    cache = load_cache(data_dir)

    # Find all ancestors
    ancestors_to_summarize = {node_id}
    changed = True

    while changed:
        changed = False
        new_ancestors = set()
        for edge in graph.get("edges", []):
            if edge.get("rel") == "contains" and edge.get("target") in ancestors_to_summarize:
                parent_id = edge.get("source")
                if parent_id not in ancestors_to_summarize:
                    new_ancestors.add(parent_id)
                    changed = True
        ancestors_to_summarize.update(new_ancestors)

    # Summarize in dependency order: children before parents
    node_children: dict[str, list[str]] = {}
    for node in graph.get("nodes", []):
        node_children[node["id"]] = []

    for edge in graph.get("edges", []):
        if edge.get("rel") == "contains":
            source = edge.get("source")
            target = edge.get("target")
            if source in node_children:
                node_children[source].append(target)

    visited = set()

    def traverse(node_id: str):
        if node_id in visited or node_id not in ancestors_to_summarize:
            return
        visited.add(node_id)

        for child_id in node_children.get(node_id, []):
            traverse(child_id)

        node = next((n for n in graph.get("nodes", []) if n["id"] == node_id), None)
        if not node:
            return

        if node_id in cache and node_id != node_id:  # Keep the originally requested node fresh
            return

        node_type = node.get("type", "unknown")

        if node_type == "function":
            prompt_input = _build_func_input(node, graph)
        else:
            prompt_input = _build_parent_input(node, graph, cache)

        if on_progress:
            on_progress(f"Summarizing {node_type}: {node['name']}")

        system_prompt = _load_prompt(f"{node_type}_describe")
        if not system_prompt:
            cache[node_id] = {
                "summary": f"[Error: no prompt for {node_type}]",
                "name": node["name"],
                "node_type": node_type,
            }
            save_cache(cache, data_dir)
            return

        summary_tokens = []
        try:
            for token in llm_client.stream_messages([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_input},
            ]):
                summary_tokens.append(token)
        except Exception as e:
            print(f"[summary] error summarizing {node_id}: {e}", file=sys.stderr)
            cache[node_id] = {
                "summary": f"[Error: {e}]",
                "name": node["name"],
                "node_type": node_type,
            }
            save_cache(cache, data_dir)
            return

        summary = "".join(summary_tokens).strip()
        cache[node_id] = {
            "summary": summary,
            "name": node["name"],
            "node_type": node_type,
        }
        save_cache(cache, data_dir)

    traverse(node_id)

    return cache


def save_tree(graph: dict[str, Any], cache: dict[str, dict], data_dir: Path) -> None:
    """Build and save the hierarchical summary tree."""
    data_dir.mkdir(parents=True, exist_ok=True)

    def build_tree_node(node_id: str) -> dict[str, Any]:
        node = next((n for n in graph.get("nodes", []) if n["id"] == node_id), None)
        if not node:
            return {}

        cache_entry = cache.get(node_id, {})
        children_ids = []
        for edge in graph.get("edges", []):
            if edge.get("rel") == "contains" and edge.get("source") == node_id:
                children_ids.append(edge.get("target"))

        children = [build_tree_node(cid) for cid in children_ids]

        return {
            "id": node_id,
            "type": node.get("type"),
            "name": node.get("name"),
            "summary": cache_entry.get("summary", ""),
            "children": [c for c in children if c],
        }

    # Find root node (repo)
    repo_node = next((n for n in graph.get("nodes", []) if n.get("type") == "repo"), None)
    if not repo_node:
        print("[summary] warning: no repo node found", file=sys.stderr)
        return

    tree = build_tree_node(repo_node["id"])

    summary_path = data_dir / "summary.json"
    tmp_path = summary_path.with_suffix(".tmp")

    json_text = json.dumps({
        "repo_name": graph.get("repo_name", "unknown"),
        "generated_at": graph.get("created_at", ""),
        "tree": tree,
    }, ensure_ascii=False, indent=2)

    tmp_path.write_text(json_text, encoding="utf-8")
    tmp_path.replace(summary_path)
    print(f"[summary] wrote {summary_path}", file=sys.stderr)
