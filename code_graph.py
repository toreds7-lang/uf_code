"""Build and manage the knowledge graph for code repositories."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from code_parser import RepoGraph, ParsedFile, ParsedClass, ParsedFunction


def _make_node_id(node_type: str, *parts: str) -> str:
    """Generate stable node ID using :: separator."""
    sanitized = [p.replace(" ", "_").replace("/", "_") for p in parts]
    return "::".join([node_type] + sanitized)


def build_graph(repo: RepoGraph, call_graph: dict[str, list[str]]) -> dict[str, Any]:
    """Build knowledge graph with nodes and edges.

    Node types: repo, folder, file, class, function
    Edge types: contains, imports, calls, inherits
    """
    nodes = []
    edges = []
    node_ids = set()

    # 1. Repo node
    repo_id = _make_node_id("repo", repo.repo_name)
    nodes.append({
        "id": repo_id,
        "type": "repo",
        "name": repo.repo_name,
        "summary": "",
        "metadata": {"root": repo.repo_name},
    })
    node_ids.add(repo_id)

    # 2. Folder nodes and contains edges
    folder_to_id = {}
    for folder_path in sorted(repo.folders.keys()):
        folder_name = folder_path.split("/")[-1] if folder_path else repo.repo_name
        folder_id = _make_node_id("folder", folder_path)
        folder_to_id[folder_path] = folder_id

        # Add folder node
        nodes.append({
            "id": folder_id,
            "type": "folder",
            "name": folder_name,
            "summary": "",
            "metadata": {"path": folder_path},
        })
        node_ids.add(folder_id)

        # Add contains edge: repo → folder or parent folder → folder
        if folder_path:
            parent_folder = "/".join(folder_path.split("/")[:-1])
            if parent_folder in folder_to_id:
                parent_id = folder_to_id[parent_folder]
            else:
                parent_id = repo_id
        else:
            parent_id = repo_id

        edges.append({
            "source": parent_id,
            "target": folder_id,
            "rel": "contains",
        })

    # 3. File, class, and function nodes
    file_to_id = {}
    class_to_id = {}
    function_to_id = {}

    for file_path, parsed_file in repo.files.items():
        # File node
        file_id = _make_node_id("file", file_path)
        file_to_id[file_path] = file_id
        file_name = Path(file_path).name

        nodes.append({
            "id": file_id,
            "type": "file",
            "name": file_name,
            "summary": "",
            "metadata": {
                "path": file_path,
                "module": parsed_file.module_name,
                "language": parsed_file.language,
                "docstring": parsed_file.docstring,
            },
        })
        node_ids.add(file_id)

        # File → folder contains edge
        folder_path = str(Path(file_path).parent)
        if folder_path == ".":
            folder_path = ""
        parent_folder_id = folder_to_id.get(folder_path, repo_id)
        edges.append({
            "source": parent_folder_id,
            "target": file_id,
            "rel": "contains",
        })

        # Class nodes
        for cls in parsed_file.classes:
            class_id = _make_node_id("class", file_path, cls.name)
            class_to_id[f"{file_path}::{cls.name}"] = class_id

            nodes.append({
                "id": class_id,
                "type": "class",
                "name": cls.name,
                "summary": "",
                "metadata": {
                    "file": file_path,
                    "bases": cls.bases,
                    "docstring": cls.docstring,
                    "lineno": cls.lineno,
                },
            })
            node_ids.add(class_id)

            # File → class contains edge
            edges.append({
                "source": file_id,
                "target": class_id,
                "rel": "contains",
            })

            # Class → method contains edges and method nodes
            for method in cls.methods:
                method_id = _make_node_id("function", file_path, cls.name, method.name)
                function_to_id[f"{file_path}::{cls.name}::{method.name}"] = method_id

                nodes.append({
                    "id": method_id,
                    "type": "function",
                    "name": method.name,
                    "qualified_name": method.qualified_name,
                    "summary": "",
                    "metadata": {
                        "file": file_path,
                        "class": cls.name,
                        "args": [{"name": a.name, "annotation": a.annotation} for a in method.args],
                        "returns": method.returns,
                        "docstring": method.docstring,
                        "decorators": method.decorators,
                        "lineno": method.lineno,
                    },
                })
                node_ids.add(method_id)

                # Class → method contains edge
                edges.append({
                    "source": class_id,
                    "target": method_id,
                    "rel": "contains",
                })

        # Top-level function nodes
        for func in parsed_file.functions:
            func_id = _make_node_id("function", file_path, func.name)
            function_to_id[f"{file_path}::{func.name}"] = func_id

            nodes.append({
                "id": func_id,
                "type": "function",
                "name": func.name,
                "qualified_name": func.qualified_name,
                "summary": "",
                "metadata": {
                    "file": file_path,
                    "args": [{"name": a.name, "annotation": a.annotation} for a in func.args],
                    "returns": func.returns,
                    "docstring": func.docstring,
                    "decorators": func.decorators,
                    "lineno": func.lineno,
                },
            })
            node_ids.add(func_id)

            # File → function contains edge
            edges.append({
                "source": file_id,
                "target": func_id,
                "rel": "contains",
            })

    # 4. Import edges (file → file)
    for file_path, parsed_file in repo.files.items():
        file_id = file_to_id[file_path]
        for imp in parsed_file.imports:
            # Try to resolve import to a repo file
            for other_file_path, other_parsed in repo.files.items():
                if other_parsed.module_name == imp.module or other_parsed.module_name.startswith(imp.module + "."):
                    target_file_id = file_to_id[other_file_path]
                    edges.append({
                        "source": file_id,
                        "target": target_file_id,
                        "rel": "imports",
                    })
                    break

    # 5. Call edges (function → function)
    for caller_id_raw, callees_raw in call_graph.items():
        # caller_id_raw is like "filepath::classname::funcname" or "filepath::funcname"
        parts = caller_id_raw.split("::")
        if len(parts) >= 3:
            file_path, class_name, func_name = parts[0], parts[1], parts[2]
            caller_id = _make_node_id("function", file_path, class_name, func_name)
        else:
            file_path, func_name = parts[0], parts[1]
            caller_id = _make_node_id("function", file_path, func_name)

        for callee_id_raw in callees_raw:
            parts = callee_id_raw.split("::")
            if len(parts) >= 3:
                file_path, class_name, func_name = parts[0], parts[1], parts[2]
                callee_id = _make_node_id("function", file_path, class_name, func_name)
            else:
                file_path, func_name = parts[0], parts[1]
                callee_id = _make_node_id("function", file_path, func_name)

            if caller_id in node_ids and callee_id in node_ids:
                edges.append({
                    "source": caller_id,
                    "target": callee_id,
                    "rel": "calls",
                })

    # 6. Inherits edges (class → base class)
    for file_path, parsed_file in repo.files.items():
        for cls in parsed_file.classes:
            class_id = _make_node_id("class", file_path, cls.name)
            for base in cls.bases:
                # Try to find the base class in the repo
                for other_file_path, other_parsed in repo.files.items():
                    for other_cls in other_parsed.classes:
                        if other_cls.name == base:
                            base_class_id = _make_node_id("class", other_file_path, other_cls.name)
                            edges.append({
                                "source": class_id,
                                "target": base_class_id,
                                "rel": "inherits",
                            })
                            break

    # Remove duplicate edges
    edges_dedup = []
    seen = set()
    for edge in edges:
        key = (edge["source"], edge["target"], edge["rel"])
        if key not in seen:
            seen.add(key)
            edges_dedup.append(edge)

    return {
        "version": 1,
        "repo_name": repo.repo_name,
        "created_at": datetime.now().isoformat(),
        "nodes": nodes,
        "edges": edges_dedup,
    }


def load_graph(data_dir: Path) -> dict[str, Any]:
    """Load graph.json from data directory."""
    graph_path = data_dir / "graph.json"
    if graph_path.exists():
        try:
            return json.loads(graph_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[graph] warning: could not load graph: {e}", file=sys.stderr)
    return {
        "version": 1,
        "repo_name": "unknown",
        "created_at": datetime.now().isoformat(),
        "nodes": [],
        "edges": [],
    }


def save_graph(graph: dict[str, Any], data_dir: Path) -> None:
    """Atomically save graph.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    graph_path = data_dir / "graph.json"
    tmp_path = graph_path.with_suffix(".tmp")

    json_text = json.dumps(graph, ensure_ascii=False, indent=2)
    tmp_path.write_text(json_text, encoding="utf-8")

    tmp_path.replace(graph_path)
    print(f"[graph] wrote {graph_path}", file=sys.stderr)
