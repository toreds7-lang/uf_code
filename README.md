# Code Repository Explanation Tool

A sophisticated tool that analyzes Python code repositories and generates:
- **Knowledge graphs** with nodes (repo, folder, file, class, function) and edges (contains, imports, calls, inherits)
- **Bottom-up LLM summaries**: function → class → file → folder → repo
- **Tool-description style function docs**: name, typed inputs/outputs, purpose, role in system
- **Interactive web viewer**: browse the graph, tree view, and ask questions via RAG-powered chat
- **Architecture reports**: human-readable Markdown summaries

## Features

✅ **Parse Python, JavaScript, TypeScript** (Python via ast module, JS/TS via optional tree-sitter)  
✅ **Extract function signatures with type hints and docstrings**  
✅ **Static call graph resolution** (three-tier best-effort across files)  
✅ **LLM-powered hierarchical summarization** with disk caching (resumable)  
✅ **Knowledge graph visualization** (vis-network)  
✅ **RAG search** over summaries for intelligent Q&A  
✅ **Works with GitHub URLs and local paths**  
✅ **Streaming API endpoints** for long-running operations  

## Installation

```bash
# Clone or navigate to the project directory
cd D:\2026_Agent\2_understanding_fast_code

# Install Python dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
cp env_example.txt env.txt
# Edit env.txt and add your OPENAI_API_KEY
```

## Quick Start

### 1. Analyze a Local Repository

```bash
python run.py D:\path\to\your\repo
```

This will:
- Parse all Python files
- Build a knowledge graph (graph.json)
- Generate LLM summaries (summary.json)
- Extract key concepts (highlights.json)
- Build a RAG index (rag.json + rag.npz)
- Generate an architecture report (report.md)

Output goes to `data/<repo_name>/`

### 2. Analyze a GitHub Repository

```bash
python run.py https://github.com/gxy-gxy/DeepRAG
```

The tool will clone the repo to a temp directory and analyze it.

### 3. Browse Results Interactively

```bash
python serve.py deeprag
```

Then open `http://localhost:8000` in your browser. You'll see:
- **Left panel**: hierarchical tree (repo → folder → file → class → function)
- **Center panel**: interactive knowledge graph (click nodes to explore)
- **Right panel**: detailed summaries and RAG-powered chat

## Processing Pipeline

The tool runs 10 steps, each resumable via caching:

```
1. repo_loader.load_repo()         → local path or git clone
2. code_parser.parse_repo()        → extract AST, no LLM
3. code_parser.resolve_call_graph()→ static cross-file call resolution
4. code_graph.build_graph()        → NetworkX → graph.json
5. code_summary.summarize_all()    → LLM summaries (incremental cache)
6. code_summary.save_tree()        → hierarchical summary.json
7. highlights.highlight_all()      → key concepts per file
8. rag.build_index()               → embed summaries, build search index
9. report.generate_report()        → architecture report.md
```

Use `--steps parse` to stop after parsing (no LLM cost).

## Command-Line Options

```bash
python run.py <source> [--force] [--steps STEP] [--filter FILE]

  source              GitHub URL or local path
  --force             Force re-clone/re-parse (bypass caches)
  --steps parse|summarize|highlights|rag|all
                      Which steps to run (default: all)
  --filter FILE       Process only a specific file (optional)
```

Examples:

```bash
# Parse only (no LLM, 30 seconds)
python run.py https://github.com/gxy-gxy/DeepRAG --steps parse

# Summarize a specific file (for iteration)
python run.py ./myrepo --steps summarize --filter src/core.py

# Force re-analysis (bypass all caches)
python run.py ./myrepo --force
```

## Output Files

Inside `data/<repo_name>/`:

| File | Purpose |
|------|---------|
| `graph.json` | Full knowledge graph: nodes (repo, folder, file, class, function) + edges (contains, imports, calls, inherits) |
| `summary.json` | Hierarchical tree with LLM summaries for each node |
| `summary_cache.json` | Flat cache of summaries (used internally, survives interruptions) |
| `highlights.json` | Key concepts extracted from each file |
| `rag.json` + `rag.npz` | Embeddings and metadata for RAG search |
| `report.md` | Human-readable architecture report |

## How It Works

### 1. Parsing (No LLM)

Uses Python's built-in `ast` module to extract:
- **Classes** with base classes and methods
- **Functions** with args (type hints), return types, docstrings, decorators
- **Imports** (both `import X` and `from X import Y`)
- **Call sites** (static detection: `foo()`, `self.bar()`, `module.baz()`)

Builds a knowledge graph with nodes and edges:
- **Nodes**: repo, folder, file, class, function
- **Edges**: contains (structural), imports, calls (resolved across files), inherits

### 2. Bottom-Up Summarization

**Post-order traversal** of the tree:
1. Summarize each **function** first (cheapest) → tool-description format:
   ```
   Function: forward
   Inputs: input_ids (Tensor), attention_mask (Tensor)
   Outputs: Tensor
   Purpose: Runs the forward pass and makes retrieval decisions.
   Role: Called during training and inference.
   ```

2. Summarize each **class** using its method summaries
3. Summarize each **file** using class + function summaries
4. Summarize each **folder** using file summaries
5. Summarize the **repo** using folder summaries

All results cached to `summary_cache.json`. If the pipeline is interrupted, rerunning picks up where it left off.

### 3. Knowledge Graph Visualization

The interactive viewer at `http://localhost:8000` renders the graph with:
- **Node colors** by type: gold (repo), blue (folder), green (file), orange (class), gray (function)
- **Edge styles** by relation: thin gray (contains), dashed blue (imports), orange arrows (calls), thick purple (inherits)
- **Filtering**: toggle edge types on/off
- **Clicking nodes**: loads detailed summaries and metadata in the right panel

### 4. RAG-Powered Chat

The chat box in the right panel:
1. Takes your question
2. Searches the RAG index (embeddings of all summaries)
3. Retrieves top-5 relevant summaries
4. Feeds them + your question to Claude
5. Streams the response

Example questions:
- "What does the DeepRAGModel class do?"
- "How are imports used in this file?"
- "What functions are called most frequently?"

## Advanced Usage

### Programmatic API

```python
from code_parser import parse_repo, resolve_call_graph
from code_graph import build_graph, save_graph
from code_summary import summarize_all, save_tree
from rag import build_index

# Parse
repo = parse_repo(Path("./myrepo"))
call_graph = resolve_call_graph(repo)

# Graph
graph = build_graph(repo, call_graph)
save_graph(graph, data_dir)

# Summaries
cache = summarize_all(graph, data_dir)
save_tree(graph, cache, data_dir)

# RAG
build_index(cache, data_dir)
```

### Custom Prompts

All prompts are plain `.txt` files in `prompts/`:
- `func_describe.system.txt` — function summarization
- `class_describe.system.txt` — class summarization
- `file_describe.system.txt` — file summarization
- `folder_describe.system.txt` — folder summarization
- `repo_describe.system.txt` — repo summarization
- `code_highlight.system.txt` — key concept extraction
- `code_qa.system.txt` — Q&A over code

Edit them to customize the tool's behavior. Changes take effect immediately on the next run (prompts are reloaded fresh).

## Configuration

Edit `env.txt`:

```bash
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini          # or gpt-4o, gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
LLM_BASE_URL=                  # empty for OpenAI, or set for vLLM compatible endpoint
EMBEDDING_BASE_URL=            # empty for OpenAI, or set for vLLM compatible endpoint
DATA_DIR=data                  # where to store output
```

## Performance

| Step | Time | Notes |
|------|------|-------|
| Parsing (50 files) | ~5 seconds | No LLM, pure AST |
| Call graph resolution | ~1 second | Static analysis |
| Graph building | ~1 second | Pure computation |
| **Summarization** | **~2-5 minutes** | LLM API calls (expensive, but cached) |
| Highlights | ~2 minutes | LLM extraction of key concepts |
| RAG indexing | ~30 seconds | Embedding API calls |
| Report generation | ~1 second | Pure formatting |

**Total first run**: ~5-10 minutes (depending on repo size).  
**Subsequent runs**: <5 seconds (all cached).  
**Rerun with `--force`**: same as first run.  

## Limitations

1. **Python-first**: Full support for Python. JavaScript/TypeScript require optional tree-sitter installation (currently disabled by default due to dependency conflicts).
2. **Static analysis**: Cannot detect runtime-injected calls or dynamic dispatch.
3. **Same-file preference**: Call resolution favors same-file matches before cross-file.
4. **Token budget**: Large repos may exceed LLM context windows; handled gracefully with chunking.
5. **No refactoring**: The tool is read-only and never modifies your code.

## Examples

### Example 1: Analyze DeepRAG

```bash
python run.py https://github.com/gxy-gxy/DeepRAG --steps parse
# Fast: 10 seconds, just shows the graph structure

python run.py https://github.com/gxy-gxy/DeepRAG
# Full pipeline: 5 minutes, includes LLM summaries, opens in viewer
```

### Example 2: Analyze Your Own Project

```bash
python run.py ./my_project
# Creates data/my_project/

python serve.py my_project
# Open http://localhost:8000
# Browse the graph, click nodes, ask questions
```

### Example 3: Iterate on Summaries

```bash
# Edit a prompt
nano prompts/func_describe.system.txt

# Re-summarize (uses cache, very fast)
python run.py ./my_project
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'langchain'"**
→ Run `pip install -r requirements.txt`

**"OPENAI_API_KEY not found"**
→ Create `env.txt` and set your API key (see env_example.txt)

**"Repo is empty after parsing"**
→ Check that the repo has `.py` files. The tool only processes Python by default (JS/TS are optional).

**"Summarization is very slow"**
→ This is expected: LLM calls take 10-30 seconds per function. Use `--steps parse` for parsing only.

**"Memory error on large repos"**
→ The tool chunks large summaries and streams them. For repos with 1000+ functions, expect higher memory usage during summarization.

## Architecture

```
D:\2026_Agent\2_understanding_fast_code\
├── config.py              # env.txt → typed constants
├── llm_client.py          # LangChain OpenAI/vLLM client
├── rag.py                 # RAG index: chunk, embed, search
├── repo_loader.py         # Git clone or local path
├── code_parser.py         # AST → ParsedFile/Class/Function
├── code_graph.py          # ParsedFile + call graph → graph.json
├── code_summary.py        # Post-order LLM summarization
├── highlights.py          # Key concept extraction
├── report.py              # Markdown report generation
├── run.py                 # CLI orchestrator
├── serve.py               # FastAPI server + endpoints
├── viewer.html            # Single-file SPA (vis-network + chat)
├── prompts/               # 7 prompt templates (.txt files)
└── data/                  # Output per repo
    └── <repo_name>/
        ├── graph.json
        ├── summary.json
        ├── summary_cache.json
        ├── highlights.json
        ├── rag.json
        ├── rag.npz
        └── report.md
```

## Design Principles

1. **Mirrors the PDF analysis tool** (`paper_read_project_pdf`) but for code:
   - Parsing → graph building → bottom-up summarization → RAG → report
   - Disk-cache pattern for all expensive operations
   - Streaming API endpoints
   - Single-file interactive viewer

2. **Best-effort static analysis**: Function calls, imports, and inheritance are resolved using three-tier fallback (same-file → cross-file → unresolved).

3. **Incremental caching**: Every node is cached independently. Interrupt anytime; rerun resumes from last cached node.

4. **Prompt-first**: All LLM behavior is controlled by prompts in `prompts/`. Edit them to customize without changing Python code.

5. **No code modification**: The tool is read-only and analyzes code as-is.

## Contributing

To extend this tool:
- **Add JS/TS support**: Install `tree-sitter` and uncomment the tree-sitter loading code in `code_parser.py`
- **Add custom prompts**: Create new `.txt` files in `prompts/` and use them in `code_summary.py`
- **Add new graph relationships**: Edit `code_graph.py` to define new edge types
- **Customize the viewer**: Edit `viewer.html` to add new UI features

## License

Use as you like. Built with reference to the PDF analysis tool at `D:\2026_Agent\1_Understanding_fast\paper_read_project_pdf\`.
