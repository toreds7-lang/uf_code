"""Generate human-readable architecture report from summary tree."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _escape_md(text: str) -> str:
    """Basic markdown escaping."""
    return text.replace("_", r"\_").replace("[", r"\[").replace("]", r"\]")


def _format_tree_section(node: dict[str, Any], depth: int = 1) -> str:
    """Recursively format a tree node as Markdown."""
    lines = []
    indent = "#" * (depth + 1)

    node_type = node.get("type", "unknown")
    node_name = node.get("name", "unknown")
    summary = node.get("summary", "")

    if node_type == "repo":
        lines.append(f"# {_escape_md(node_name)}")
    else:
        lines.append(f"{indent} {_escape_md(node_name)} ({node_type})")

    if summary:
        lines.append("")
        lines.append(summary)
        lines.append("")

    # Recursively format children
    for child in node.get("children", []):
        child_type = child.get("type", "unknown")
        if child_type != "function":  # Skip functions in report for brevity
            lines.append(_format_tree_section(child, depth + 1))

    return "\n".join(lines)


def generate_report(
    summary_tree: dict[str, Any],
    highlights: dict[str, dict],
    data_dir: Path,
) -> None:
    """Generate report.md from summary tree and highlights."""
    data_dir.mkdir(parents=True, exist_ok=True)

    lines = []

    # Title and repo overview
    repo_name = summary_tree.get("repo_name", "Unknown Repository")
    generated_at = summary_tree.get("generated_at", "")

    lines.append(f"# {repo_name}")
    lines.append("")
    if generated_at:
        lines.append(f"*Generated: {generated_at}*")
        lines.append("")

    # Tree structure
    tree_node = summary_tree.get("tree", {})
    if tree_node:
        tree_summary = tree_node.get("summary", "")
        if tree_summary:
            lines.append("## Overview")
            lines.append("")
            lines.append(tree_summary)
            lines.append("")

        # Recursively add children
        lines.append("## Architecture")
        lines.append("")
        for child in tree_node.get("children", []):
            lines.append(_format_tree_section(child, depth=1))

    # Key concepts summary
    if highlights:
        lines.append("\n## Key Concepts by Module")
        lines.append("")

        for file_path in sorted(highlights.keys()):
            concepts = highlights[file_path].get("concepts", [])
            if concepts:
                file_name = Path(file_path).name
                lines.append(f"### {file_name}")
                lines.append("")
                for concept in concepts[:10]:  # Top 10
                    lines.append(f"- `{concept}`")
                lines.append("")

    report_text = "\n".join(lines)

    # Save report
    report_path = data_dir / "report.md"
    tmp_path = report_path.with_suffix(".tmp")
    tmp_path.write_text(report_text, encoding="utf-8")
    tmp_path.replace(report_path)

    print(f"[report] wrote {report_path}", file=sys.stderr)
