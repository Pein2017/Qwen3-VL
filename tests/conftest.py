from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_first() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    return repo_root


def _evict_src_if_from_other_checkout(repo_root: Path) -> None:
    expected_src_dir = repo_root / "src"
    src_mod = sys.modules.get("src")
    if src_mod is None:
        return

    src_paths = getattr(src_mod, "__path__", None)
    if not src_paths:
        return

    expected = str(expected_src_dir)
    if any(str(p).startswith(expected) for p in src_paths):
        return

    for name in list(sys.modules.keys()):
        if name == "src" or name.startswith("src."):
            sys.modules.pop(name, None)


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    repo_root = _ensure_repo_root_first()
    _evict_src_if_from_other_checkout(repo_root)
