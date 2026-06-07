#!/usr/bin/env python3
"""Main CLI orchestrator: repo URL/path → full analysis pipeline."""
import argparse
import json
import sys
from pathlib import Path

import code_graph
import code_parser
import code_summary
import highlights
import rag
import reconstruct
import report
import repo_loader
from config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Analyze a code repository and generate knowledge graph + summaries.")
    parser.add_argument("source", help="GitHub URL (https://github.com/...) or local path")
    parser.add_argument("--force", action="store_true", help="Force re-clone/re-parse (bypass caches)")
    parser.add_argument("--steps", default="all", help="Which steps to run (all, parse, summarize, highlights, rag, report, reconstruct)")
    parser.add_argument("--filter", default="", help="Filter to specific file/folder (optional)")

    args = parser.parse_args()

    print(f"[run] starting pipeline for: {args.source}", file=sys.stderr)

    # Step 1: Load repository
    try:
        repo_path = repo_loader.load_repo(args.source, force_reclone=args.force)
    except Exception as e:
        print(f"[run] ERROR: could not load repo: {e}", file=sys.stderr)
        return 1

    repo_name = repo_path.name
    data_dir = DATA_DIR / repo_name
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "source_root.txt").write_text(str(repo_path), encoding="utf-8")

    # Step 2: Parse repository (always run, but cached)
    print(f"[run] step 1: parsing repository", file=sys.stderr)
    repo = code_parser.parse_repo(repo_path)

    # Step 3: Resolve call graph
    print(f"[run] step 2: resolving call graph", file=sys.stderr)
    call_graph = code_parser.resolve_call_graph(repo)

    # Step 4: Build knowledge graph
    print(f"[run] step 3: building knowledge graph", file=sys.stderr)
    graph = code_graph.build_graph(repo, call_graph)
    code_graph.save_graph(graph, data_dir)

    if args.steps in ("parse",):
        print(f"[run] stopping after parse step", file=sys.stderr)
        return 0

    # Reconstruct-only: source-grounded reverse-engineering blueprint (skips
    # the describe/highlights/rag passes; needs only the graph + original source).
    if args.steps == "reconstruct":
        print(f"[run] step R: reverse-engineering reconstruction blueprint", file=sys.stderr)
        recon_cache = reconstruct.reconstruct_all(graph, repo_path, data_dir)
        reconstruct.save_markdown(graph, recon_cache, data_dir)
        print(f"[run] stopping after reconstruct step", file=sys.stderr)
        print(f"[run] output directory: {data_dir}", file=sys.stderr)
        return 0

    # Step 5: Summarize all nodes (bottom-up)
    print(f"[run] step 4: summarizing nodes (this may take a while...)", file=sys.stderr)
    cache = code_summary.summarize_all(graph, data_dir)

    # Step 6: Save summary tree
    print(f"[run] step 5: building summary tree", file=sys.stderr)
    code_summary.save_tree(graph, cache, data_dir)

    if args.steps in ("summarize", "parse"):
        print(f"[run] stopping after summarize step", file=sys.stderr)
        return 0

    # Step 7: Extract highlights (key concepts per file)
    print(f"[run] step 6: extracting key concepts", file=sys.stderr)
    file_sources = {}
    for file_path, parsed_file in repo.files.items():
        try:
            full_path = repo_path / file_path
            file_sources[file_path] = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[run] warning: could not read {file_path}: {e}", file=sys.stderr)

    highlights_cache = highlights.highlight_all(file_sources, data_dir)

    if args.steps in ("highlights",):
        print(f"[run] stopping after highlights step", file=sys.stderr)
        return 0

    # Step 8: Build RAG index
    print(f"[run] step 7: building RAG index", file=sys.stderr)
    # Need to enrich cache with node_type and name
    enhanced_cache = {}
    for node in graph.get("nodes", []):
        node_id = node["id"]
        if node_id in cache:
            enhanced_cache[node_id] = {
                **cache[node_id],
                "node_type": node.get("type"),
                "name": node.get("name"),
            }
    rag.build_index(enhanced_cache, data_dir)

    if args.steps in ("rag",):
        print(f"[run] stopping after rag step", file=sys.stderr)
        return 0

    # Step 9: Generate report
    print(f"[run] step 8: generating report", file=sys.stderr)
    summary_tree_path = data_dir / "summary.json"
    if summary_tree_path.exists():
        summary_data = json.loads(summary_tree_path.read_text(encoding="utf-8"))
        report.generate_report(summary_data, highlights_cache, data_dir)

    # Step 10: Reverse-engineering reconstruction blueprint (source-grounded)
    print(f"[run] step 9: reconstruction blueprint", file=sys.stderr)
    recon_cache = reconstruct.reconstruct_all(graph, repo_path, data_dir)
    reconstruct.save_markdown(graph, recon_cache, data_dir)

    print(f"[run] SUCCESS: analysis complete!", file=sys.stderr)
    print(f"[run] output directory: {data_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
