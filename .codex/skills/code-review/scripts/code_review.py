from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


@dataclass(frozen=True)
class Issue:
    tool: str
    kind: str
    severity: str
    code: str | None
    message: str
    path: str | None = None
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    category: str | None = None
    clause: str | None = None
    remediation: str | None = None


@dataclass(frozen=True)
class ToolRun:
    argv: Sequence[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CodeReviewOptions:
    paths: tuple[str, ...]
    mode: str
    fix: bool
    format: bool
    format_check: bool
    include_tool_runs: bool
    include_issues: bool
    exit_with_status: bool
    write_artifacts: bool
    output_dir: str
    max_issues: int

    def __post_init__(self) -> None:
        if self.mode not in {"fast", "full"}:
            raise ValueError("options.mode must be one of: fast, full")
        if self.max_issues < 0:
            raise ValueError("options.max_issues must be >= 0")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CodeReviewOptions":
        if not isinstance(raw, MappingABC):
            raise TypeError("options must be a mapping")

        def get_bool(key: str) -> bool:
            value = raw.get(key)
            if not isinstance(value, bool):
                raise TypeError(f"options.{key} must be bool")
            return value

        def get_str(key: str) -> str:
            value = raw.get(key)
            if not isinstance(value, str):
                raise TypeError(f"options.{key} must be str")
            if value == "":
                raise ValueError(f"options.{key} must be non-empty")
            return value

        raw_paths_obj = raw.get("paths")
        if not isinstance(raw_paths_obj, Sequence) or isinstance(
            raw_paths_obj, (str, bytes)
        ):
            raise TypeError("options.paths must be a sequence of strings")
        raw_paths = cast(Sequence[object], raw_paths_obj)
        paths: list[str] = []
        for idx, item in enumerate(raw_paths):
            if not isinstance(item, str):
                raise TypeError(f"options.paths[{idx}] must be str")
            if item == "":
                raise ValueError(f"options.paths[{idx}] must be non-empty")
            paths.append(item)

        mode = raw.get("mode")
        if not isinstance(mode, str):
            raise TypeError("options.mode must be str")

        max_issues = raw.get("max_issues")
        if not isinstance(max_issues, int):
            raise TypeError("options.max_issues must be int")

        return cls(
            paths=tuple(paths),
            mode=mode,
            fix=get_bool("fix"),
            format=get_bool("format"),
            format_check=get_bool("format_check"),
            include_tool_runs=get_bool("include_tool_runs"),
            include_issues=get_bool("include_issues"),
            exit_with_status=get_bool("exit_with_status"),
            write_artifacts=get_bool("write_artifacts"),
            output_dir=get_str("output_dir"),
            max_issues=max_issues,
        )


RUFF_PREFIX_TO_CATEGORY: Mapping[str, str] = {
    "E": "Pycodestyle",
    "W": "Pycodestyle",
    "F": "Pyflakes",
    "I": "Imports",
    "UP": "Modernization",
    "B": "Bugbear",
    "SIM": "Simplify",
    "C4": "Comprehensions",
    "PIE": "Misc",
    "RUF": "Ruff",
    "N": "Naming",
    "ARG": "Unused arguments",
    "ERA": "Dead code",
    "PL": "Pylint",
    "TRY": "Exceptions",
    "PERF": "Performance",
}


def _print_err(message: str) -> None:
    sys.stderr.write(message.rstrip() + "\n")


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyrightconfig.json").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    return start.resolve()


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _run(argv: Sequence[str], cwd: Path) -> ToolRun:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return ToolRun(
        argv=tuple(argv),
        cwd=str(cwd),
        exit_code=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mapping = cast(MappingABC[object, object], payload)
    for key in mapping.keys():
        if not isinstance(key, str):
            return None
    return cast(dict[str, Any], mapping)


@contextmanager
def _scoped_pyright_project(
    *,
    repo_root: Path,
    include_paths: Sequence[str],
    output_dir: Path | None,
) -> Any:
    """
    Create a temporary pyright config that scopes `include` to the requested paths.

    Reason: basedpyright/pyright will ignore file arguments if the project config's
    include/exclude doesn't cover them. A scoped config ensures the requested paths
    are actually analyzed while still preserving the project's diagnostic settings.
    """

    base_config_path = repo_root / "pyrightconfig.json"
    base = _safe_load_json(base_config_path)
    if base is None:
        yield None
        return

    scoped: dict[str, Any] = {**base, "include": list(include_paths)}

    if output_dir is not None:
        scoped_path = output_dir / "pyrightconfig.scoped.json"
        _write_json(scoped_path, scoped)
        yield scoped_path
        return

    with tempfile.TemporaryDirectory(prefix="code_review_pyright_") as tmp_dir:
        scoped_path = Path(tmp_dir) / "pyrightconfig.scoped.json"
        _write_json(scoped_path, scoped)
        yield scoped_path


def _as_str_keyed_mapping(raw: object) -> Mapping[str, object] | None:
    if not isinstance(raw, MappingABC):
        return None
    mapping = cast(MappingABC[object, object], raw)
    for key in mapping.keys():
        if not isinstance(key, str):
            return None
    return cast(Mapping[str, object], mapping)


def _collect_python_files(paths: Sequence[Path]) -> list[Path]:
    skip_dir_names = {
        ".codex",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "model_cache",
        "output",
        "output_post",
        "tb",
        "tmp",
        "archive",
        "backup",
        "old",
    }

    files: list[Path] = []
    for root in paths:
        if root.is_file():
            if root.suffix == ".py":
                files.append(root)
            continue

        if not root.is_dir():
            continue

        for candidate in root.rglob("*.py"):
            if any(part in skip_dir_names for part in candidate.parts):
                continue
            files.append(candidate)

    # Stable ordering for deterministic reports
    return sorted({p.resolve() for p in files}, key=lambda p: str(p))


def _ruff_category(code: str | None, kind: str) -> str:
    if kind == "format":
        return "Format"
    if not code:
        return "Unknown"

    m = re.match(r"^[A-Z]+", code)
    prefix = m.group(0) if m else code
    return RUFF_PREFIX_TO_CATEGORY.get(prefix, prefix)


def _parse_ruff_check_json(repo_root: Path, stdout: str) -> list[Issue]:
    try:
        payload = json.loads(stdout or "[]")
    except json.JSONDecodeError as e:
        return [
            Issue(
                tool="ruff",
                kind="lint",
                severity="error",
                code=None,
                message=f"Failed to parse ruff JSON output: {e}",
            )
        ]

    if not isinstance(payload, list):
        return [
            Issue(
                tool="ruff",
                kind="lint",
                severity="error",
                code=None,
                message=f"Unexpected ruff JSON payload type: {type(payload).__name__}",
            )
        ]

    issues: list[Issue] = []
    for raw_item in cast(list[object], payload):
        item = _as_str_keyed_mapping(raw_item)
        if item is None:
            continue

        code_obj = item.get("code")
        code = str(code_obj) if code_obj is not None else None
        message_obj = item.get("message")
        message = str(message_obj) if message_obj is not None else ""
        filename_obj = item.get("filename")
        filename = str(filename_obj) if filename_obj is not None else None

        location = _as_str_keyed_mapping(item.get("location"))
        end_location = _as_str_keyed_mapping(item.get("end_location"))

        row_obj = location.get("row") if location is not None else None
        col_obj = location.get("column") if location is not None else None
        end_row_obj = end_location.get("row") if end_location is not None else None
        end_col_obj = end_location.get("column") if end_location is not None else None

        category = _ruff_category(str(code) if code else None, kind="lint")

        issues.append(
            Issue(
                tool="ruff",
                kind="lint",
                severity="error",
                code=code,
                message=message,
                path=filename,
                line=int(row_obj) if isinstance(row_obj, int) else None,
                column=int(col_obj) if isinstance(col_obj, int) else None,
                end_line=int(end_row_obj) if isinstance(end_row_obj, int) else None,
                end_column=int(end_col_obj) if isinstance(end_col_obj, int) else None,
                category=category,
                remediation="Run `ruff check --fix` where safe, then address remaining diagnostics.",
            )
        )

    # Ruff includes paths relative to the invocation cwd; normalize to repo-relative when possible.
    normalized: list[Issue] = []
    for issue in issues:
        if issue.path is None:
            normalized.append(issue)
            continue
        normalized.append(
            replace(issue, path=_relpath(repo_root, (repo_root / issue.path).resolve()))
        )
    return normalized


def _parse_ruff_format_output(repo_root: Path, stdout: str) -> list[Issue]:
    # Ruff format output is not JSON; normalize from stable prefix lines.
    # Example: "Would reformat: path/to/file.py"
    issues: list[Issue] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Would reformat:"):
            continue
        path_part = line.removeprefix("Would reformat:").strip()
        issues.append(
            Issue(
                tool="ruff",
                kind="format",
                severity="warning",
                code="RUFF-FORMAT",
                message="File is not formatted according to ruff formatter.",
                path=_relpath(repo_root, repo_root / path_part),
                category="Format",
                remediation="Run `ruff format` on the affected paths.",
            )
        )
    return issues


def _pick_pyright_tool() -> str | None:
    if shutil.which("pyright"):
        return "pyright"
    if shutil.which("basedpyright"):
        return "basedpyright"
    return None


def _pyright_include_paths(repo_root: Path, raw_paths: Sequence[str]) -> list[str]:
    include: list[str] = []
    for p in raw_paths:
        path = Path(p)
        resolved = path if path.is_absolute() else (repo_root / path)
        # NOTE: Pyright/basedpyright resolve "include" entries relative to the
        # *config file location*, not the current working directory. Since we
        # write a scoped config file into a temp/artifacts directory, using
        # repo-relative paths like "src" would incorrectly resolve to
        # "tmp/code-review/src" and result in 0 files analyzed.
        #
        # Use absolute paths to make the scoped config location-independent.
        include.append(str(resolved.resolve()))
    # Preserve ordering but remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in include:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _parse_pyright_json(repo_root: Path, tool_name: str, stdout: str) -> list[Issue]:
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError as e:
        return [
            Issue(
                tool=tool_name,
                kind="type",
                severity="error",
                code=None,
                message=f"Failed to parse {tool_name} JSON output: {e}",
            )
        ]

    payload_map = _as_str_keyed_mapping(payload)
    if payload_map is None:
        return [
            Issue(
                tool=tool_name,
                kind="type",
                severity="error",
                code=None,
                message=(
                    f"Unexpected {tool_name} JSON payload type: {type(payload).__name__}"
                ),
            )
        ]

    diagnostics_obj = payload_map.get("generalDiagnostics")
    diagnostics: object = diagnostics_obj if diagnostics_obj is not None else []
    if not isinstance(diagnostics, list):
        return [
            Issue(
                tool=tool_name,
                kind="type",
                severity="error",
                code=None,
                message=f"Unexpected {tool_name} JSON shape: generalDiagnostics is not a list",
            )
        ]

    issues: list[Issue] = []
    for raw_diag in cast(list[object], diagnostics):
        diag = _as_str_keyed_mapping(raw_diag)
        if diag is None:
            continue
        file_path_obj = diag.get("file")
        file_path = str(file_path_obj) if isinstance(file_path_obj, str) else None
        severity_obj = diag.get("severity")
        severity = str(severity_obj) if severity_obj is not None else "error"
        message_obj = diag.get("message")
        message = str(message_obj) if message_obj is not None else ""
        rule_obj = diag.get("rule")
        rule = str(rule_obj) if rule_obj is not None else None

        range_map = _as_str_keyed_mapping(diag.get("range"))
        start_map = (
            _as_str_keyed_mapping(range_map.get("start"))
            if range_map is not None
            else None
        )
        end_map = (
            _as_str_keyed_mapping(range_map.get("end"))
            if range_map is not None
            else None
        )

        # Pyright JSON uses 0-based line/character (LSP style); normalize to 1-based.
        line_obj = start_map.get("line") if start_map is not None else None
        col_obj = start_map.get("character") if start_map is not None else None
        end_line_obj = end_map.get("line") if end_map is not None else None
        end_col_obj = end_map.get("character") if end_map is not None else None

        rel_file = None
        if file_path is not None:
            rel_file = _relpath(repo_root, Path(file_path))

        issues.append(
            Issue(
                tool=tool_name,
                kind="type",
                severity=str(severity),
                code=rule,
                message=message,
                path=rel_file,
                line=int(line_obj) + 1 if isinstance(line_obj, int) else None,
                column=int(col_obj) + 1 if isinstance(col_obj, int) else None,
                end_line=int(end_line_obj) + 1
                if isinstance(end_line_obj, int)
                else None,
                end_column=int(end_col_obj) + 1
                if isinstance(end_col_obj, int)
                else None,
                category="Type checking",
                remediation="Fix type errors/warnings; avoid `Any` at boundaries; add structured schemas where needed.",
            )
        )

    return issues


def _pyright_files_analyzed(stdout: str) -> int | None:
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return None
    payload_map = _as_str_keyed_mapping(payload)
    if payload_map is None:
        return None
    summary_map = _as_str_keyed_mapping(payload_map.get("summary"))
    if summary_map is None:
        return None
    files_obj = summary_map.get("filesAnalyzed")
    if not isinstance(files_obj, int):
        return None
    return int(files_obj)


def _ast_contains_name(node: ast.AST, *, names: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
        if isinstance(child, ast.Attribute) and child.attr in names:
            return True
    return False


def _is_suspicious_mapping_annotation(annotation: ast.AST) -> bool:
    # Heuristic: flag dict/list-like annotations that likely represent non-trivial structures,
    # especially those involving Any or unparameterized collections.
    if isinstance(annotation, ast.Name) and annotation.id in {
        "dict",
        "list",
        "Dict",
        "List",
    }:
        return True

    if isinstance(annotation, ast.Subscript):
        value = annotation.value
        if isinstance(value, ast.Name):
            base = value.id
            if base in {"dict", "Dict", "Mapping", "MutableMapping"}:
                return _ast_contains_name(annotation, names={"Any", "object"})
            if base in {"list", "List", "Sequence"}:
                return _ast_contains_name(annotation, names={"Any", "object"})

    return _ast_contains_name(annotation, names={"Any"}) and _ast_contains_name(
        annotation,
        names={
            "dict",
            "list",
            "Dict",
            "List",
            "Mapping",
            "MutableMapping",
            "Sequence",
        },
    )


def _constitution_scan_file(repo_root: Path, path: Path) -> list[Issue]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        return [
            Issue(
                tool="constitution",
                kind="constitution",
                severity="warning",
                code="SCHEMA-READ-FAILED",
                message=f"Failed to read file for constitution scan: {e}",
                path=_relpath(repo_root, path),
                clause="Purpose",
                remediation="Ensure files are readable and encoded as UTF-8.",
            )
        ]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [
            Issue(
                tool="constitution",
                kind="constitution",
                severity="warning",
                code="SCHEMA-PARSE-FAILED",
                message=f"Failed to parse Python AST for constitution scan: {e.msg}",
                path=_relpath(repo_root, path),
                line=e.lineno,
                column=e.offset,
                clause="Purpose",
                remediation="Fix syntax errors before running schema audits.",
            )
        ]

    issues: list[Issue] = []

    # 1) Non-trivial mapping/list usage in function signatures or returns (heuristic).
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        signature_nodes: list[tuple[str, ast.arg, ast.AST]] = []
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if arg.annotation is not None:
                signature_nodes.append(("param", arg, arg.annotation))

        if node.args.vararg and node.args.vararg.annotation is not None:
            signature_nodes.append(
                ("vararg", node.args.vararg, node.args.vararg.annotation)
            )
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            signature_nodes.append(
                ("kwarg", node.args.kwarg, node.args.kwarg.annotation)
            )
        if node.returns is not None:
            signature_nodes.append(("return", ast.arg(arg="return"), node.returns))

        for role, arg, annotation in signature_nodes:
            if not _is_suspicious_mapping_annotation(annotation):
                continue
            arg_name = getattr(arg, "arg", "<unknown>")
            col_offset = getattr(node, "col_offset", None)
            issues.append(
                Issue(
                    tool="constitution",
                    kind="constitution",
                    severity="warning",
                    code="SCHEMA-NONTRIVIAL-MAPPING",
                    message=(
                        "Potential non-trivial mapping/list type used in function signature; "
                        "Schema Constitution prefers structured types (dataclass/TypedDict/Pydantic) at boundaries."
                    ),
                    path=_relpath(repo_root, path),
                    line=getattr(node, "lineno", None),
                    column=col_offset + 1 if isinstance(col_offset, int) else None,
                    category="Schema modeling",
                    clause="Function signatures and returns",
                    remediation=(
                        f"Replace `{arg_name}` {role} annotation with a structured type (dataclass/TypedDict) "
                        "and validate at boundary; avoid `Any` in semantic contracts."
                    ),
                )
            )

    # 2) dataclass without frozen=True (heuristic).
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        dataclass_decorators: list[ast.AST] = []
        for deco in node.decorator_list:
            if isinstance(deco, ast.Name) and deco.id == "dataclass":
                dataclass_decorators.append(deco)
            elif isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
                dataclass_decorators.append(deco)
            elif isinstance(deco, ast.Call):
                func = deco.func
                if (isinstance(func, ast.Name) and func.id == "dataclass") or (
                    isinstance(func, ast.Attribute) and func.attr == "dataclass"
                ):
                    dataclass_decorators.append(deco)

        if not dataclass_decorators:
            continue

        frozen = None
        for deco in dataclass_decorators:
            if not isinstance(deco, ast.Call):
                frozen = False
                break
            for kw in deco.keywords:
                if kw.arg == "frozen":
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, bool
                    ):
                        frozen = kw.value.value
                    else:
                        frozen = None

        if frozen is False:
            col_offset = getattr(node, "col_offset", None)
            issues.append(
                Issue(
                    tool="constitution",
                    kind="constitution",
                    severity="info",
                    code="SCHEMA-DATACLASS-NOT-FROZEN",
                    message="dataclass detected without `frozen=True`; Schema Constitution prefers `dataclass(frozen=True)` for internal structured state.",
                    path=_relpath(repo_root, path),
                    line=getattr(node, "lineno", None),
                    column=col_offset + 1 if isinstance(col_offset, int) else None,
                    category="Schema modeling",
                    clause="Type selection rules",
                    remediation="Use `@dataclass(frozen=True)` for config/state records; validate in `__post_init__` and parse external mappings via `from_mapping`.",
                )
            )

    # 3) cast(...) usage (heuristic).
    cast_calls = 0
    first_cast_loc: tuple[int | None, int | None] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "cast":
            cast_calls += 1
        elif isinstance(func, ast.Attribute) and func.attr == "cast":
            cast_calls += 1
        if cast_calls == 1:
            first_cast_loc = (
                getattr(node, "lineno", None),
                getattr(node, "col_offset", None),
            )

    if cast_calls:
        lineno, col = first_cast_loc if first_cast_loc is not None else (None, None)
        issues.append(
            Issue(
                tool="constitution",
                kind="constitution",
                severity="info",
                code="SCHEMA-CAST-USED",
                message=f"`cast(...)` used ({cast_calls} call(s)); ensure validation occurs before casting and that casts are not used to bypass schema modeling.",
                path=_relpath(repo_root, path),
                line=lineno,
                column=col + 1 if isinstance(col, int) else None,
                category="Schema modeling",
                clause="Type selection rules",
                remediation="Validate raw mappings at entry (TypeError/ValueError with full field path) before applying `cast(...)`.",
            )
        )

    return issues


def _architecture_import_graph(
    repo_root: Path, files: Sequence[Path]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    # Returns (graph: module -> module imports), and module->path mapping.
    path_by_module: dict[str, str] = {}
    module_by_path: dict[str, str] = {}
    for path in files:
        rel = Path(_relpath(repo_root, path))
        if rel.suffix != ".py":
            continue
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        module = ".".join(parts)
        path_by_module[module] = str(rel)
        module_by_path[str(rel)] = module

    modules = set(path_by_module.keys())

    def resolve_module_to_internal(module: str) -> str | None:
        if module in modules:
            return module
        candidate_py = repo_root / Path(module.replace(".", "/") + ".py")
        candidate_init = repo_root / Path(module.replace(".", "/") + "/__init__.py")
        if candidate_py.exists():
            cand = Path(_relpath(repo_root, candidate_py))
            return module_by_path.get(str(cand))
        if candidate_init.exists():
            cand = Path(_relpath(repo_root, candidate_init))
            return module_by_path.get(str(cand))
        return None

    graph: dict[str, set[str]] = {m: set() for m in modules}

    for path in files:
        rel_path_str = _relpath(repo_root, path)
        module = module_by_path.get(rel_path_str)
        if not module:
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            continue

        # Determine current package for relative imports.
        package_parts = module.split(".")[:-1]
        if rel_path_str.endswith("__init__.py"):
            package_parts = module.split(".")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    resolved = resolve_module_to_internal(imported)
                    if resolved:
                        graph[module].add(resolved)

            if isinstance(node, ast.ImportFrom):
                level = int(node.level or 0)
                base_parts = list(package_parts)
                if level > 1:
                    drop = min(len(base_parts), level - 1)
                    base_parts = base_parts[: len(base_parts) - drop]
                base = ".".join(base_parts)
                if node.module:
                    base = f"{base}.{node.module}" if base else node.module

                base_resolved = resolve_module_to_internal(base) if base else None
                if base_resolved:
                    graph[module].add(base_resolved)

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    resolved = resolve_module_to_internal(candidate)
                    if resolved:
                        graph[module].add(resolved)

    return graph, path_by_module


def _find_cycles(graph: Mapping[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            # Cycle found; slice stack from first occurrence.
            try:
                idx = stack.index(node)
            except ValueError:
                idx = 0
            cycle = stack[idx:] + [node]
            cycles.append(cycle)
            return

        visiting.add(node)
        stack.append(node)
        for nxt in sorted(graph.get(node, set())):
            dfs(nxt)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph.keys()):
        dfs(node)

    # Deduplicate cycles by canonical rotation.
    canonical: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        body = cycle[:-1]
        rotations = [tuple(body[i:] + body[:i]) for i in range(len(body))]
        key = min(rotations)
        if key in canonical:
            continue
        canonical.add(key)
        unique.append(body + [body[0]])
    return unique


def _architecture_issues(repo_root: Path, files: Sequence[Path]) -> list[Issue]:
    graph, path_by_module = _architecture_import_graph(repo_root, files)
    cycles = _find_cycles(graph)

    issues: list[Issue] = []
    for cycle in cycles:
        # Render as module chain; point to first module file as anchor.
        anchor = cycle[0]
        anchor_path = path_by_module.get(anchor)
        issues.append(
            Issue(
                tool="architecture",
                kind="architecture",
                severity="warning",
                code="IMPORT-CYCLE",
                message="Internal import cycle detected: " + " -> ".join(cycle),
                path=anchor_path,
                category="Coupling",
                remediation="Break the cycle by extracting shared interfaces/types into a lower-level module or by inverting dependencies.",
            )
        )

    fanout = {m: len(deps) for m, deps in graph.items()}
    if fanout:
        top = sorted(fanout.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for module, count in top:
            if count < 25:
                continue
            issues.append(
                Issue(
                    tool="architecture",
                    kind="architecture",
                    severity="info",
                    code="HIGH-FANOUT",
                    message=f"High internal import fan-out ({count}) suggests tight coupling; consider splitting responsibilities or introducing facades.",
                    path=path_by_module.get(module),
                    category="Coupling",
                    remediation="Extract cohesive submodules; prefer dependency inversion for high-level orchestration code.",
                )
            )

    return issues


def _status_from_issues(issues: Sequence[Issue]) -> str:
    severity_counts = Counter(i.severity for i in issues)
    if severity_counts.get("error", 0) > 0:
        return "Fail"
    if severity_counts.get("warning", 0) > 0:
        return "Conditional Pass"
    return "Pass"


_SEVERITY_ORDER: Mapping[str, int] = {
    "error": 0,
    "warning": 1,
    "information": 2,
    "info": 2,
}


def _issue_sort_key(issue: Issue) -> tuple[int, str, str, int, int, str]:
    severity_rank = _SEVERITY_ORDER.get(issue.severity, 99)
    path = issue.path or ""
    line = issue.line or 0
    column = issue.column or 0
    code = issue.code or ""
    return (severity_rank, issue.tool, path, line, column, code)


def _sorted_issues(issues: Sequence[Issue]) -> list[Issue]:
    return sorted(issues, key=_issue_sort_key)


def _exit_code_from_status(status: str) -> int:
    # Conventional-ish mapping:
    # - Pass -> 0 (success)
    # - Conditional Pass -> 1 (warnings)
    # - Fail -> 2 (errors)
    if status == "Pass":
        return 0
    if status == "Conditional Pass":
        return 1
    return 2


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: int(v) for k, v in counter.most_common()}


def _counts_by_tool_severity(issues: Sequence[Issue]) -> dict[str, dict[str, int]]:
    by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    for issue in issues:
        by_tool[issue.tool][issue.severity] += 1
    return {tool: _counter_to_dict(counts) for tool, counts in sorted(by_tool.items())}


def _counts_where(
    issues: Sequence[Issue], *, tool: str, key: str, default: str
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for issue in issues:
        if issue.tool != tool:
            continue
        value = getattr(issue, key)
        counter[str(value) if value else default] += 1
    return _counter_to_dict(counter)


def _fix_plan() -> list[str]:
    return [
        "Fix format: run `ruff format` on the reviewed paths.",
        "Fix lint: run `ruff check --fix` where safe; address remaining errors.",
        "Fix typing: resolve pyright errors first, then warnings; prefer structured schemas over `Any` at boundaries.",
        "Address Schema Constitution suspects: replace non-trivial mappings/lists in public interfaces with dataclasses/TypedDict + validation.",
        "Reduce coupling: break import cycles and split high fan-out modules into cohesive units.",
    ]


def _build_json_report(
    *,
    issues: Sequence[Issue],
    tool_runs: Sequence[ToolRun],
    max_issues: int,
    mode: str,
    artifacts_dir: str | None,
    include_tool_runs: bool,
    include_issues: bool,
    applied_fixes: Mapping[str, bool],
    pyright_files_analyzed: int | None,
) -> dict[str, Any]:
    status = _status_from_issues(issues)
    exit_code = _exit_code_from_status(status)
    sorted_issues = _sorted_issues(issues)

    shown = sorted_issues if max_issues <= 0 else sorted_issues[:max_issues]
    truncated = max(0, len(sorted_issues) - len(shown))

    pyright_tools = {"pyright", "basedpyright"}
    ruff_counts = _counts_where(issues, tool="ruff", key="category", default="Unknown")
    pyright_sev: Counter[str] = Counter()
    for issue in issues:
        if issue.tool in pyright_tools:
            pyright_sev[issue.severity] += 1

    constitution_counts = _counts_where(
        issues, tool="constitution", key="clause", default="Unspecified clause"
    )
    architecture_counts = _counts_where(
        issues, tool="architecture", key="code", default="UNSPECIFIED"
    )

    schema_attention = sum(constitution_counts.values()) > 0
    architecture_attention = sum(architecture_counts.values()) > 0
    typecheck_error_attention = int(pyright_sev.get("error", 0)) > 0
    typecheck_skipped = pyright_files_analyzed == 0

    report: dict[str, Any] = {
        "summary": {
            "result": status,
            "exit_code": exit_code,
            "mode": mode,
            "risk_by_tool": _counts_by_tool_severity(issues),
            "total_issues": len(sorted_issues),
            "artifacts_dir": artifacts_dir,
            "applied_fixes": dict(applied_fixes),
        },
        "static_analysis_findings": {
            "ruff": {
                "by_category": ruff_counts,
                "total": sum(ruff_counts.values()),
            },
            "pyright": {
                "by_severity": _counter_to_dict(pyright_sev),
                "total": int(sum(pyright_sev.values())),
                "files_analyzed": pyright_files_analyzed,
            },
        },
        "constitution_violations": {
            "enabled": mode == "full",
            "by_clause": constitution_counts,
            "total": sum(constitution_counts.values()),
        },
        "design_and_architecture_review": {
            "enabled": mode == "full",
            "automated_findings": architecture_counts,
            "total": sum(architecture_counts.values()),
            "notes": (
                "Architecture/design assessment is evidence-based; automate import-cycle/fan-out detection and add manual review for API boundaries, dependency direction, extensibility, and testability."
            ),
        },
        "attention_required": {
            "schema_or_nontrivial": bool(schema_attention),
            "heavy_entanglement": bool(architecture_attention),
            "type_check_errors": bool(typecheck_error_attention),
            "type_check_skipped": bool(typecheck_skipped),
            "action": (
                "If non-trivial findings (schema/architecture/type errors) are present, decide whether to skip for now or refactor immediately."
            ),
        },
        "actionable_fix_plan": _fix_plan(),
    }
    if include_tool_runs:
        report["tool_runs"] = [
            {"argv": list(run.argv), "cwd": run.cwd, "exit_code": run.exit_code}
            for run in tool_runs
        ]
    if include_issues:
        report["issues"] = [asdict(i) for i in shown]
        report["issues_truncated"] = truncated
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-review Python code: apply simple ruff fixes/formatting, run type checking, "
            "and emit a lightweight JSON report."
        )
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["src"],
        help="Paths (files or directories) to review. Default: src",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="full",
        help="fast: ruff + type checking only. full: also run constitution + architecture heuristics.",
    )
    parser.add_argument(
        "--fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply safe auto-fixes via `ruff check --fix` before collecting remaining lint issues.",
    )
    parser.add_argument(
        "--format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply formatting via `ruff format` before collecting remaining issues.",
    )
    parser.add_argument(
        "--format-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run `ruff format --check` (only when --no-format) to report formatter drift without writing changes.",
    )
    parser.add_argument(
        "--include-tool-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include command argv/cwd/exit codes in the JSON output (normally hidden).",
    )
    parser.add_argument(
        "--include-issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include individual issues inline in stdout JSON (truncated by --max-issues).",
    )
    parser.add_argument(
        "--exit-with-status",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exit non-zero based on result (Pass=0, Conditional Pass=1, Fail=2). "
            "Default is to always exit 0 to keep stdout JSON clean when wrapped by tools like `conda run`."
        ),
    )
    parser.add_argument(
        "--write-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write JSON artifacts (issues.json, summary.json, tool raw outputs) to --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/code-review",
        help="Directory to write artifacts when --write-artifacts is enabled. Default: tmp/code-review",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=200,
        help="Max issues included inline in stdout JSON (<=0 means no limit).",
    )

    namespace = parser.parse_args(argv)
    options = CodeReviewOptions.from_mapping(vars(namespace))

    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(script_path)

    paths = [
        Path(p) if Path(p).is_absolute() else (repo_root / p) for p in options.paths
    ]

    output_dir: Path | None = None
    if options.write_artifacts:
        output_dir_path = (
            Path(options.output_dir)
            if Path(options.output_dir).is_absolute()
            else (repo_root / options.output_dir)
        )
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir_path

    issues: list[Issue] = []
    tool_runs: list[ToolRun] = []
    applied_fixes: dict[str, bool] = {"format": False, "ruff_fix": False}
    pyright_files_analyzed: int | None = None

    if not shutil.which("ruff"):
        _print_err(
            "ruff not found on PATH. Expected `conda run -n ms ruff ...` environment."
        )
        issues.append(
            Issue(
                tool="ruff",
                kind="lint",
                severity="error",
                code="RUFF-NOT-FOUND",
                message="ruff not found on PATH; cannot run lint/format.",
                remediation="Install ruff or run via `conda run -n ms ...` in the project environment.",
            )
        )
    else:
        format_issues: list[Issue] = []
        if options.format:
            ruff_format_apply = _run(
                ["ruff", "format", *[str(p) for p in options.paths]],
                cwd=repo_root,
            )
            tool_runs.append(ruff_format_apply)
            applied_fixes["format"] = ruff_format_apply.exit_code == 0
        elif options.format_check:
            ruff_format_check = _run(
                ["ruff", "format", "--check", *[str(p) for p in options.paths]],
                cwd=repo_root,
            )
            tool_runs.append(ruff_format_check)
            format_issues = _parse_ruff_format_output(
                repo_root,
                "\n".join([ruff_format_check.stdout, ruff_format_check.stderr]).strip(),
            )

        if output_dir is not None:
            _write_json(
                output_dir / "ruff_format.json", [asdict(i) for i in format_issues]
            )
        issues.extend(format_issues)

        if options.fix:
            ruff_fix = _run(
                ["ruff", "check", "--fix", *[str(p) for p in options.paths]],
                cwd=repo_root,
            )
            tool_runs.append(ruff_fix)
            applied_fixes["ruff_fix"] = True

        ruff_check = _run(
            [
                "ruff",
                "check",
                "--output-format",
                "json",
                *[str(p) for p in options.paths],
            ],
            cwd=repo_root,
        )
        tool_runs.append(ruff_check)
        if output_dir is not None:
            _write_text(output_dir / "ruff_check.json", ruff_check.stdout)
        issues.extend(_parse_ruff_check_json(repo_root, ruff_check.stdout))

    pyright_tool = _pick_pyright_tool()
    if not pyright_tool:
        _print_err(
            "pyright/basedpyright not found on PATH. Type checking will be skipped."
        )
        issues.append(
            Issue(
                tool="pyright",
                kind="type",
                severity="warning",
                code="PYRIGHT-NOT-FOUND",
                message="pyright-compatible type checker not found (pyright or basedpyright).",
                remediation="Install pyright (npm/pip) or basedpyright, then re-run.",
            )
        )
    else:
        include_paths = _pyright_include_paths(repo_root, options.paths)
        with _scoped_pyright_project(
            repo_root=repo_root, include_paths=include_paths, output_dir=output_dir
        ) as project_config:
            if project_config is None:
                pyright_argv = [
                    pyright_tool,
                    "--outputjson",
                    *[str(p) for p in options.paths],
                ]
            else:
                pyright_argv = [
                    pyright_tool,
                    "--outputjson",
                    "-p",
                    str(project_config),
                ]
            pyright_run = _run(pyright_argv, cwd=repo_root)
            tool_runs.append(pyright_run)
            pyright_files_analyzed = _pyright_files_analyzed(pyright_run.stdout)
            if pyright_files_analyzed == 0:
                issues.append(
                    Issue(
                        tool=pyright_tool,
                        kind="type",
                        severity="warning",
                        code="PYRIGHT-NO-FILES-ANALYZED",
                        message=(
                            "Type checker analyzed 0 files for the requested paths "
                            "(hidden directories like `.codex/` may be skipped)."
                        ),
                        category="Type checking",
                        remediation=(
                            "Move code under a non-hidden directory for type checking, "
                            "or adjust the type checker configuration to include hidden paths."
                        ),
                    )
                )
            if output_dir is not None:
                _write_text(output_dir / "pyright.json", pyright_run.stdout)
            issues.extend(
                _parse_pyright_json(repo_root, pyright_tool, pyright_run.stdout)
            )

    python_files: list[Path] | None = None

    if options.mode == "full":
        python_files = _collect_python_files(paths)

        constitution_doc = repo_root / "docs/reference/SCHEMA_CONSTITUTION.md"
        if not constitution_doc.exists():
            issues.append(
                Issue(
                    tool="constitution",
                    kind="constitution",
                    severity="warning",
                    code="SCHEMA-CONSTITUTION-MISSING",
                    message="Schema Constitution doc not found at docs/reference/SCHEMA_CONSTITUTION.md; skipping compliance audit.",
                )
            )
        else:
            for py_path in python_files:
                issues.extend(_constitution_scan_file(repo_root, py_path))

        issues.extend(_architecture_issues(repo_root, python_files))

    json_report = _build_json_report(
        issues=issues,
        tool_runs=tool_runs,
        max_issues=int(options.max_issues),
        mode=str(options.mode),
        artifacts_dir=str(output_dir) if output_dir is not None else None,
        include_tool_runs=bool(options.include_tool_runs),
        include_issues=bool(options.include_issues),
        applied_fixes=applied_fixes,
        pyright_files_analyzed=pyright_files_analyzed,
    )

    if output_dir is not None:
        _write_json(
            output_dir / "issues.json", [asdict(i) for i in _sorted_issues(issues)]
        )
        _write_json(output_dir / "summary.json", json_report)

    print(json.dumps(json_report, ensure_ascii=False, indent=2))
    if options.exit_with_status:
        return int(json_report["summary"]["exit_code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
