"""AST-based code parsing using Python's ast module and optional tree-sitter support."""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional tree-sitter support
try:
    from py_tree_sitter_languages import get_language, get_parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


# Language detection
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
}

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".next"}


@dataclass
class ArgInfo:
    name: str
    annotation: str | None = None


@dataclass
class ParsedFunction:
    name: str
    qualified_name: str
    file_path: str
    lineno: int
    args: list[ArgInfo] = field(default_factory=list)
    returns: str | None = None
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    is_method: bool = False
    class_name: str | None = None


@dataclass
class ParsedClass:
    name: str
    file_path: str
    lineno: int
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None
    methods: list[ParsedFunction] = field(default_factory=list)


@dataclass
class ImportInfo:
    module: str
    names: list[str] = field(default_factory=list)
    is_from: bool = False


@dataclass
class ParsedFile:
    path: str
    language: str
    module_name: str
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    functions: list[ParsedFunction] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class RepoGraph:
    repo_name: str
    root_path: str
    files: dict[str, ParsedFile] = field(default_factory=dict)
    folders: dict[str, list[str]] = field(default_factory=dict)


class CodeParser:
    """Parses Python code using ast module (primary) with optional tree-sitter fallback for JS/TS."""

    def __init__(self):
        self.parsers: dict[str, Any] = {}
        self.languages: dict[str, Any] = {}

        if HAS_TREE_SITTER:
            for lang in ["javascript", "typescript"]:
                try:
                    self.languages[lang] = get_language(lang)
                    self.parsers[lang] = get_parser(lang)
                except Exception as e:
                    print(f"[parser] warning: could not load {lang} grammar: {e}", file=sys.stderr)

    def parse_file(self, path: Path, repo_root: Path) -> ParsedFile | None:
        """Parse a single file and extract functions, classes, imports."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[parser] warning: could not read {path}: {e}", file=sys.stderr)
            return None

        ext = path.suffix.lower()
        language = LANG_MAP.get(ext)
        if not language:
            return None

        rel_path = path.relative_to(repo_root).as_posix()
        module_name = self._path_to_module(rel_path, language)

        parsed_file = ParsedFile(
            path=rel_path,
            language=language,
            module_name=module_name,
        )

        if language == "python":
            self._parse_python_ast(content, parsed_file, path)
        elif language in ["javascript", "typescript"] and HAS_TREE_SITTER:
            parser = self.parsers.get(language)
            if parser:
                try:
                    tree = parser.parse(content.encode("utf-8"))
                    self._parse_js_ts(tree, content, parsed_file, path, language)
                except Exception as e:
                    print(f"[parser] warning: parse failed for {path}: {e}", file=sys.stderr)
                    return None

        return parsed_file

    def _path_to_module(self, rel_path: str, language: str) -> str:
        """Convert file path to module name."""
        parts = rel_path.replace("\\", "/").split("/")
        if parts[-1] == "index.py" or parts[-1] == "index.js":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].rsplit(".", 1)[0]
        return ".".join(p for p in parts if p and not p.startswith("."))

    def _parse_python_ast(self, content: str, parsed_file: ParsedFile, path: Path) -> None:
        """Extract Python functions, classes, imports using ast module."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"[parser] warning: syntax error in {path}: {e}", file=sys.stderr)
            return

        # Extract module docstring
        if (tree.body and isinstance(tree.body[0], ast.Expr) and
            isinstance(tree.body[0].value, ast.Constant) and
            isinstance(tree.body[0].value.value, str)):
            parsed_file.docstring = tree.body[0].value.value

        # Walk AST
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls = self._parse_python_class_ast(node, path)
                if cls:
                    parsed_file.classes.append(cls)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._parse_python_function_ast(node, path)
                if func:
                    parsed_file.functions.append(func)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imp = self._parse_python_import_ast(node)
                if imp:
                    parsed_file.imports.append(imp)

    def _parse_python_class_ast(self, node: ast.ClassDef, path: Path) -> ParsedClass | None:
        """Extract a Python class using ast."""
        bases = [ast.unparse(base) for base in node.bases] if hasattr(ast, 'unparse') else []
        docstring = ast.get_docstring(node)
        methods = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._parse_python_function_ast(item, path, class_name=node.name)
                if method:
                    methods.append(method)

        return ParsedClass(
            name=node.name,
            file_path=path.name,
            lineno=node.lineno,
            bases=bases,
            docstring=docstring,
            methods=methods,
        )

    def _parse_python_function_ast(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: Path,
        class_name: str | None = None,
    ) -> ParsedFunction | None:
        """Extract a Python function using ast."""
        qualified_name = f"{class_name}.{node.name}" if class_name else node.name

        # Extract arguments with annotations
        args = []
        for arg in node.args.args:
            annotation = None
            if arg.annotation and hasattr(ast, 'unparse'):
                annotation = ast.unparse(arg.annotation)
            args.append(ArgInfo(name=arg.arg, annotation=annotation))

        # Extract return type
        returns = None
        if node.returns and hasattr(ast, 'unparse'):
            returns = ast.unparse(node.returns)

        # Extract decorators
        decorators = []
        if hasattr(ast, 'unparse'):
            decorators = [ast.unparse(d) for d in node.decorator_list]

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract calls
        calls = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Name):
                    calls.append(subnode.func.id)
                elif isinstance(subnode.func, ast.Attribute):
                    if hasattr(ast, 'unparse'):
                        calls.append(ast.unparse(subnode.func))

        return ParsedFunction(
            name=node.name,
            qualified_name=qualified_name,
            file_path=path.name,
            lineno=node.lineno,
            args=args,
            returns=returns,
            docstring=docstring,
            decorators=decorators,
            calls=list(set(calls)),
            is_method=class_name is not None,
            class_name=class_name,
        )

    def _parse_python_import_ast(self, node: ast.Import | ast.ImportFrom) -> ImportInfo | None:
        """Extract an import statement using ast."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                return ImportInfo(module=alias.name, is_from=False)
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            return ImportInfo(module=node.module or "", names=names, is_from=True)
        return None

    def _parse_python_class(self, node: Any, content: str, lines: list[str], path: Path) -> ParsedClass | None:
        """Extract a Python class and its methods."""
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break

        if not name_node:
            return None

        class_name = self._get_text(name_node, content)
        lineno = node.start_point[0] + 1

        # Extract base classes
        bases = []
        for child in node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type == "identifier":
                        bases.append(self._get_text(arg, content))

        # Extract docstring
        docstring = None
        for child in node.children:
            if child.type == "block" or child.type == "suite":
                for subchild in child.children:
                    if subchild.type == "expression_statement":
                        if subchild.child_count > 0 and subchild.child(0).type == "string":
                            docstring = self._get_text(subchild.child(0), content).strip('"\'')
                            break

        # Extract methods
        methods = []
        for child in node.children:
            if child.type in ("block", "suite"):
                for method_node in child.children:
                    if method_node.type == "function_definition":
                        method = self._parse_python_function(method_node, content, lines, path, class_name)
                        if method:
                            methods.append(method)

        return ParsedClass(
            name=class_name,
            file_path=path.name,
            lineno=lineno,
            bases=bases,
            docstring=docstring,
            methods=methods,
        )

    def _parse_python_function(
        self,
        node: Any,
        content: str,
        lines: list[str],
        path: Path,
        class_name: str | None = None,
    ) -> ParsedFunction | None:
        """Extract a Python function or method."""
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break

        if not name_node:
            return None

        func_name = self._get_text(name_node, content)
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name
        lineno = node.start_point[0] + 1

        # Extract decorators
        decorators = []
        prev = node.prev_sibling
        while prev:
            if prev.type == "decorator":
                decorators.append(self._get_text(prev, content).lstrip("@"))
            prev = prev.prev_sibling

        # Extract args and return type
        args = []
        returns = None
        for child in node.children:
            if child.type == "parameters":
                args = self._extract_python_args(child, content)
            elif child.type == "type_annotation":
                returns = self._get_text(child, content)

        # Extract docstring (first string in function body)
        docstring = None
        for child in node.children:
            if child.type in ("block", "suite"):
                for subchild in child.children:
                    if subchild.type == "expression_statement":
                        if subchild.child_count > 0 and subchild.child(0).type == "string":
                            docstring = self._get_text(subchild.child(0), content).strip('"\'')
                            break
                break

        # Extract calls
        calls = []
        for child in node.children:
            if child.type in ("block", "suite"):
                calls = self._extract_python_calls(child, content)
                break

        return ParsedFunction(
            name=func_name,
            qualified_name=qualified_name,
            file_path=path.name,
            lineno=lineno,
            args=args,
            returns=returns,
            docstring=docstring,
            decorators=decorators,
            calls=calls,
            is_method=class_name is not None,
            class_name=class_name,
        )

    def _extract_python_args(self, params_node: Any, content: str) -> list[ArgInfo]:
        """Extract function parameters with type annotations."""
        args = []
        for child in params_node.children:
            if child.type == "identifier":
                args.append(ArgInfo(name=self._get_text(child, content)))
            elif child.type == "typed_parameter":
                name_part = None
                annotation_part = None
                for subchild in child.children:
                    if subchild.type == "identifier":
                        name_part = self._get_text(subchild, content)
                    elif subchild.type == "type_annotation":
                        annotation_part = self._get_text(subchild, content).lstrip(": ")
                if name_part:
                    args.append(ArgInfo(name=name_part, annotation=annotation_part))
        return args

    def _extract_python_calls(self, node: Any, content: str) -> list[str]:
        """Extract function calls from a block."""
        calls = []

        def walk_calls(n: Any):
            if n.type == "call":
                for child in n.children:
                    if child.type == "attribute":
                        call_name = self._get_text(child, content)
                        if call_name and "." in call_name:
                            calls.append(call_name)
                    elif child.type == "identifier":
                        call_name = self._get_text(child, content)
                        if call_name:
                            calls.append(call_name)
            for child in n.children:
                walk_calls(child)

        walk_calls(node)
        return list(set(calls))  # deduplicate

    def _parse_python_import(self, node: Any, content: str) -> ImportInfo | None:
        """Extract import statement."""
        text = self._get_text(node, content)
        if text.startswith("from "):
            parts = text.split()
            if len(parts) >= 4:
                module = parts[1]
                names = [p.strip(",") for p in parts[3:]]
                return ImportInfo(module=module, names=names, is_from=True)
        elif text.startswith("import "):
            module = text.replace("import ", "").split(",")[0].strip()
            return ImportInfo(module=module, is_from=False)
        return None

    def _parse_js_ts(
        self,
        tree: Any,
        content: str,
        parsed_file: ParsedFile,
        path: Path,
        language: str,
    ) -> None:
        """Extract JS/TS functions, classes, imports."""
        root = tree.root_node

        for child in root.children:
            if child.type == "class_declaration":
                cls = self._parse_js_ts_class(child, content, path)
                if cls:
                    parsed_file.classes.append(cls)
            elif child.type == "function_declaration":
                func = self._parse_js_ts_function(child, content, path)
                if func:
                    parsed_file.functions.append(func)
            elif child.type in ("import_statement", "import_specifier"):
                imp = self._parse_js_ts_import(child, content)
                if imp:
                    parsed_file.imports.append(imp)

    def _parse_js_ts_class(self, node: Any, content: str, path: Path) -> ParsedClass | None:
        """Extract a JS/TS class."""
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break

        if not name_node:
            return None

        class_name = self._get_text(name_node, content)
        lineno = node.start_point[0] + 1

        # Extract base class
        bases = []
        for child in node.children:
            if child.type == "class_heritage":
                heritage_text = self._get_text(child, content)
                if "extends" in heritage_text:
                    base = heritage_text.replace("extends", "").strip()
                    bases.append(base)

        # Extract methods
        methods = []
        for child in node.children:
            if child.type == "class_body":
                for method_node in child.children:
                    if method_node.type == "method_definition":
                        method = self._parse_js_ts_method(method_node, content, path, class_name)
                        if method:
                            methods.append(method)

        return ParsedClass(
            name=class_name,
            file_path=path.name,
            lineno=lineno,
            bases=bases,
            methods=methods,
        )

    def _parse_js_ts_function(self, node: Any, content: str, path: Path) -> ParsedFunction | None:
        """Extract a JS/TS function."""
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break

        if not name_node:
            return None

        func_name = self._get_text(name_node, content)
        lineno = node.start_point[0] + 1

        # Extract args
        args = []
        for child in node.children:
            if child.type == "formal_parameters":
                args = self._extract_js_ts_args(child, content)

        # Extract calls
        calls = []
        for child in node.children:
            if child.type == "statement_block":
                calls = self._extract_js_ts_calls(child, content)

        return ParsedFunction(
            name=func_name,
            qualified_name=func_name,
            file_path=path.name,
            lineno=lineno,
            args=args,
            calls=calls,
        )

    def _parse_js_ts_method(
        self,
        node: Any,
        content: str,
        path: Path,
        class_name: str,
    ) -> ParsedFunction | None:
        """Extract a JS/TS method."""
        name_node = None
        for child in node.children:
            if child.type == "identifier" or child.type == "property_identifier":
                name_node = child
                break

        if not name_node:
            return None

        method_name = self._get_text(name_node, content)
        qualified_name = f"{class_name}.{method_name}"
        lineno = node.start_point[0] + 1

        args = []
        for child in node.children:
            if child.type == "formal_parameters":
                args = self._extract_js_ts_args(child, content)

        calls = []
        for child in node.children:
            if child.type == "statement_block":
                calls = self._extract_js_ts_calls(child, content)

        return ParsedFunction(
            name=method_name,
            qualified_name=qualified_name,
            file_path=path.name,
            lineno=lineno,
            args=args,
            calls=calls,
            is_method=True,
            class_name=class_name,
        )

    def _extract_js_ts_args(self, node: Any, content: str) -> list[ArgInfo]:
        """Extract JS/TS function parameters."""
        args = []
        for child in node.children:
            if child.type == "identifier" or child.type == "optional_parameter":
                name = self._get_text(child, content).split(":")[0].strip()
                if name:
                    args.append(ArgInfo(name=name))
        return args

    def _extract_js_ts_calls(self, node: Any, content: str) -> list[str]:
        """Extract function calls from JS/TS block."""
        calls = []

        def walk_calls(n: Any):
            if n.type == "call_expression":
                for child in n.children:
                    if child.type == "identifier" or child.type == "member_expression":
                        call_name = self._get_text(child, content)
                        if call_name:
                            calls.append(call_name)
            for child in n.children:
                walk_calls(child)

        walk_calls(node)
        return list(set(calls))

    def _parse_js_ts_import(self, node: Any, content: str) -> ImportInfo | None:
        """Extract JS/TS import statement."""
        text = self._get_text(node, content)
        if "from" in text:
            parts = text.split("from")
            if len(parts) == 2:
                module = parts[1].strip().strip("'\"")
                return ImportInfo(module=module, is_from=True)
        return None

    def _get_text(self, node: Any, content: str) -> str:
        """Extract text from a tree-sitter node."""
        try:
            start = node.start_byte
            end = node.end_byte
            return content[start:end]
        except Exception:
            return ""


def parse_repo(root: Path) -> RepoGraph:
    """Parse all supported code files in a repository."""
    repo_name = root.name
    parser = CodeParser()

    repo_graph = RepoGraph(repo_name=repo_name, root_path=str(root))
    folders: dict[str, set[str]] = {}

    # Walk all files
    for file_path in root.rglob("*"):
        if file_path.is_file():
            # Skip unwanted dirs
            parts = file_path.relative_to(root).parts
            if any(p in SKIP_DIRS for p in parts):
                continue

            ext = file_path.suffix.lower()
            if ext not in LANG_MAP:
                continue

            parsed = parser.parse_file(file_path, root)
            if parsed:
                repo_graph.files[parsed.path] = parsed

                # Track folder hierarchy
                folder = str(file_path.parent.relative_to(root))
                if folder == ".":
                    folder = ""
                if folder:
                    folders.setdefault(folder, set()).add(parsed.path)

    # Build folder hierarchy
    for folder, items in folders.items():
        repo_graph.folders[folder] = sorted(list(items))

    print(f"[parser] parsed {len(repo_graph.files)} files from {repo_name}", file=sys.stderr)
    return repo_graph


def resolve_call_graph(repo: RepoGraph) -> dict[str, list[str]]:
    """Three-tier best-effort static call resolution within the repo."""
    call_graph: dict[str, list[str]] = {}

    # Build lookup indices
    name_to_ids: dict[str, list[str]] = {}
    qualified_to_id: dict[str, str] = {}
    module_to_file: dict[str, str] = {}

    for file_path, parsed_file in repo.files.items():
        module_to_file[parsed_file.module_name] = file_path

        for func in parsed_file.functions:
            func_id = f"{file_path}::{func.name}"
            name_to_ids.setdefault(func.name, []).append(func_id)
            qualified_to_id[func.qualified_name] = func_id

        for cls in parsed_file.classes:
            for method in cls.methods:
                method_id = f"{file_path}::{cls.name}::{method.name}"
                name_to_ids.setdefault(method.name, []).append(method_id)
                qualified_to_id[method.qualified_name] = method_id

    # Resolve calls
    for file_path, parsed_file in repo.files.items():
        for func in parsed_file.functions:
            func_id = f"{file_path}::{func.name}"
            resolved_calls = _resolve_calls(func.calls, func_id, file_path, parsed_file, name_to_ids, qualified_to_id)
            if resolved_calls:
                call_graph[func_id] = resolved_calls

        for cls in parsed_file.classes:
            for method in cls.methods:
                method_id = f"{file_path}::{cls.name}::{method.name}"
                resolved_calls = _resolve_calls(method.calls, method_id, file_path, parsed_file, name_to_ids, qualified_to_id, cls.name)
                if resolved_calls:
                    call_graph[method_id] = resolved_calls

    print(f"[parser] resolved {len(call_graph)} calls", file=sys.stderr)
    return call_graph


def _resolve_calls(
    raw_calls: list[str],
    caller_id: str,
    file_path: str,
    parsed_file: ParsedFile,
    name_to_ids: dict[str, list[str]],
    qualified_to_id: dict[str, str],
    class_name: str | None = None,
) -> list[str]:
    """Resolve raw call names to node IDs using three-tier fallback."""
    resolved = []

    for call in raw_calls:
        # Tier 1: qualified match (e.g., "ClassName.method")
        if call in qualified_to_id:
            resolved.append(qualified_to_id[call])
            continue

        # Tier 2: direct name match (same file)
        if call in name_to_ids and name_to_ids[call]:
            ids = name_to_ids[call]
            same_file = [i for i in ids if i.startswith(file_path)]
            if same_file:
                resolved.append(same_file[0])
                continue

        # Tier 3: self.method within same class
        if class_name and "self." in call:
            method_name = call.replace("self.", "")
            qualified = f"{class_name}.{method_name}"
            if qualified in qualified_to_id:
                resolved.append(qualified_to_id[qualified])
                continue

    return list(set(resolved))
