#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AST-based import boundary checks for the repo.

This script enforces lightweight dependency direction constraints introduced by
the `src/` import-surface refactor (see `tests/test_import_smoke.py`).

Checks:
1) Foundation packages MUST NOT depend on pipeline modules:
   - src.utils/** MUST NOT import src.stage_a, src.stage_b, src.sft, src.trainers, src.training
   - src.generation/** MUST NOT import src.stage_a, src.stage_b, src.sft, src.trainers, src.training
2) Cross-domain private helper imports are forbidden:
   - a module under src/<domain>/... MUST NOT import underscore-prefixed names
     from another src/<other_domain>/... module.

Exit code:
  0 when checks pass; 1 when violations are found.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

_BANNED_PIPELINE_PREFIXES: tuple[str, ...] = (
    "src.stage_a",
    "src.stage_b",
    "src.sft",
    "src.trainers",
    "src.training",
)


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    message: str


def _iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        # Skip typical cache dirs just in case.
        if "__pycache__" in path.parts:
            continue
        yield path


def _module_path_for_file(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_parts_for_file_module(file_module: str) -> list[str]:
    parts = file_module.split(".")
    # For a module `a.b.c`, package is `a.b`.
    return parts[:-1] if parts else []


def _resolve_from_import(*, file_module: str, node: ast.ImportFrom) -> str | None:
    # Absolute import.
    if node.level == 0:
        return node.module

    # Relative import: resolve against the *package* of the current module.
    pkg_parts = _package_parts_for_file_module(file_module)
    up = node.level - 1
    if up > len(pkg_parts):
        # Invalid/ambiguous; treat as unresolved.
        return None
    base_parts = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        return ".".join(base_parts + node.module.split("."))
    return ".".join(base_parts) if base_parts else None


def _has_banned_prefix(module_name: str) -> bool:
    return any(
        module_name == pfx or module_name.startswith(pfx + ".")
        for pfx in _BANNED_PIPELINE_PREFIXES
    )


def _top_level_domain(module_name: str) -> str | None:
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "src":
        return parts[1]
    return None


def _check_foundation_imports(path: Path) -> list[Violation]:
    file_module = _module_path_for_file(path)
    violations: list[Violation] = []

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if imported and _has_banned_prefix(imported):
                    violations.append(
                        Violation(
                            path=path,
                            lineno=getattr(node, "lineno", 1),
                            message=f"Forbidden import in foundation package: {imported}",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from_import(file_module=file_module, node=node)
            if resolved and _has_banned_prefix(resolved):
                violations.append(
                    Violation(
                        path=path,
                        lineno=getattr(node, "lineno", 1),
                        message=f"Forbidden import in foundation package: {resolved}",
                    )
                )

    return violations


def _check_cross_domain_private_imports(path: Path) -> list[Violation]:
    file_module = _module_path_for_file(path)
    current_domain = _top_level_domain(file_module)
    if current_domain is None:
        return []

    violations: list[Violation] = []
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = _resolve_from_import(file_module=file_module, node=node)
        if not resolved:
            continue
        imported_domain = _top_level_domain(resolved)
        if imported_domain is None or imported_domain == current_domain:
            continue
        for alias in node.names:
            name = alias.name
            if name.startswith("_"):
                violations.append(
                    Violation(
                        path=path,
                        lineno=getattr(node, "lineno", 1),
                        message=(
                            "Cross-domain private import is forbidden: "
                            f"{file_module} imports {name} from {resolved}"
                        ),
                    )
                )

    return violations


def main(argv: list[str] | None = None) -> int:
    _ = argv
    if not SRC_ROOT.exists():
        print(f"ERROR: src/ not found at {SRC_ROOT}", file=sys.stderr)
        return 2

    violations: list[Violation] = []

    # 1) Dependency direction for foundation packages.
    for sub in ("utils", "generation"):
        root = SRC_ROOT / sub
        if not root.exists():
            continue
        for path in _iter_py_files(root):
            violations.extend(_check_foundation_imports(path))

    # 2) Cross-domain private helper imports for all src/**.
    for path in _iter_py_files(SRC_ROOT):
        violations.extend(_check_cross_domain_private_imports(path))

    if violations:
        for v in violations:
            rel = v.path.relative_to(REPO_ROOT)
            print(f"{rel}:{v.lineno}: {v.message}", file=sys.stderr)
        print(f"Found {len(violations)} import-boundary violation(s).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
