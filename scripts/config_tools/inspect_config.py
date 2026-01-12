#!/usr/bin/env python3
"""
Extends-aware inspection + diff tooling for YAML training configs.

This script does NOT introduce a new config system. It uses the existing
`ConfigLoader.load_yaml_with_extends()` behavior and simply makes the merge order
and final resolved config easy to inspect and compare.

Examples:
  conda run -n ms python scripts/config_tools/inspect_config.py inspect --config configs/train/sft/dense_1024.yaml
  conda run -n ms python scripts/config_tools/inspect_config.py diff --left configs/train/sft/dense_1024.yaml --right configs/train/sft/dense_2048.yaml --profile parity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep output deterministic/diff-friendly by suppressing noisy INFO logs from ms-swift.
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("swift").setLevel(logging.WARNING)

from src.config import ConfigLoader  # noqa: E402


ALLOWED_PARITY_DIFF_PATHS: frozenset[str] = frozenset(
    {
        # Run identity / filesystem outputs
        "training.output_dir",
        "training.logging_dir",
        "training.run_name",
        "custom.dump_conversation_text",
        "custom.dump_conversation_path",
        # Telemetry/logging verbosity
        "training.logging_steps",
        "training.logging_first_step",
        "training.log_level",
        "training.log_level_replica",
        "training.report_to",
        # Checkpoint/eval scheduling
        "training.save_strategy",
        "training.save_steps",
        "training.save_total_limit",
        "training.save_last_epoch",
        "training.eval_strategy",
        "training.eval_steps",
        "training.metric_for_best_model",
        "training.greater_is_better",
    }
)

_AUGMENTATION_RELATED_DIFF_PREFIXES: tuple[str, ...] = (
    "custom.augmentation",
    "custom.augmentation_curriculum",
)


def _is_augmentation_related_diff_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}.")
        for prefix in _AUGMENTATION_RELATED_DIFF_PREFIXES
    )


def _augmentation_enabled(resolved_config: object) -> bool:
    """
    Mirrors the effective logic in `src/sft.py`:
      - If `custom.augmentation` is missing/None => disabled.
      - If `custom.augmentation` is a mapping => enabled defaults to True unless `enabled: false`.

    This is used only for parity diff classification (allowed vs forbidden).
    """

    if not isinstance(resolved_config, Mapping):
        # Be conservative: if we cannot reason about the config shape, treat augmentation as enabled.
        return True
    root_map = cast(Mapping[object, object], resolved_config)
    custom = root_map.get("custom")
    if not isinstance(custom, Mapping):
        return False
    custom_map = cast(Mapping[object, object], custom)
    aug_cfg = custom_map.get("augmentation")
    if aug_cfg is None:
        return False
    if not isinstance(aug_cfg, Mapping):
        return True
    enabled = cast(Mapping[object, object], aug_cfg).get("enabled", True)
    return bool(enabled)


def _resolve_user_path(path: Path) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config not found: {resolved}")
        return resolved

    cwd_candidate = path.resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.is_file():
        return repo_candidate

    raise FileNotFoundError(
        f"Config not found: {path} (cwd={Path.cwd()}, repo_root={REPO_ROOT})"
    )


def _format_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path.resolve())


def _normalize_extends(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        seq = cast(Sequence[object], value)
        return [str(v) for v in seq]
    return [str(value)]


def _extends_chain(config_path: Path, _visited: set[str] | None = None) -> list[Path]:
    abs_path = str(config_path.resolve())
    visited = set(_visited or set())
    if abs_path in visited:
        raise ValueError(f"Cyclic config inheritance detected at: {abs_path}")
    visited.add(abs_path)

    config = ConfigLoader.load_yaml(abs_path) or {}
    if not isinstance(config, Mapping):
        raise TypeError(f"Config root must be a mapping: {abs_path}")
    config_map = cast(Mapping[str, object], config)

    if "inherit" in config_map:
        raise ValueError(
            "Config inheritance uses 'extends'; 'inherit' is not supported."
        )

    base_refs = _normalize_extends(config_map.get("extends"))
    chain: list[Path] = []
    current_dir = Path(abs_path).parent
    for base_ref in base_refs:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (current_dir / base_path).resolve()
        chain.extend(_extends_chain(base_path, visited))
    chain.append(Path(abs_path))
    return chain


def _to_canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _format_diff_value(value: object, *, max_chars: int = 200) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = repr(value)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def _collect_diffs(
    left: object, right: object, path: str
) -> list[tuple[str, object, object]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_map = {str(k): v for k, v in cast(Mapping[object, object], left).items()}
        right_map = {str(k): v for k, v in cast(Mapping[object, object], right).items()}
        diffs: list[tuple[str, object, object]] = []
        keys = sorted(set(left_map.keys()) | set(right_map.keys()))
        for key in keys:
            next_path = f"{path}.{key}" if path else key
            if key not in left_map:
                diffs.append((next_path, None, right_map[key]))
                continue
            if key not in right_map:
                diffs.append((next_path, left_map[key], None))
                continue
            diffs.extend(_collect_diffs(left_map[key], right_map[key], next_path))
        return diffs

    is_left_seq = isinstance(left, Sequence) and not isinstance(left, (str, bytes))
    is_right_seq = isinstance(right, Sequence) and not isinstance(right, (str, bytes))
    if is_left_seq and is_right_seq:
        return [] if left == right else [(path, left, right)]

    return [] if left == right else [(path, left, right)]


def _print_chain(label: str, chain: list[Path]) -> None:
    print(f"{label} extends chain (low → high precedence):")
    for idx, path in enumerate(chain, start=1):
        print(f"  {idx:>2}. {_format_path(path)}")


def _cmd_inspect(*, config_path: Path, output_format: str) -> int:
    chain = _extends_chain(config_path)
    _print_chain("Resolved", chain)
    print("")
    resolved = ConfigLoader.load_yaml_with_extends(str(config_path))

    if output_format == "json":
        print(_to_canonical_json(resolved))
        return 0

    if output_format == "yaml":
        import yaml

        print(yaml.safe_dump(resolved, sort_keys=True, allow_unicode=True))
        return 0

    raise ValueError(f"Unsupported --format: {output_format}")


def _cmd_diff(*, left_path: Path, right_path: Path, profile: str) -> int:
    left_chain = _extends_chain(left_path)
    right_chain = _extends_chain(right_path)
    _print_chain("Left", left_chain)
    _print_chain("Right", right_chain)
    print("")

    left_resolved = ConfigLoader.load_yaml_with_extends(str(left_path))
    right_resolved = ConfigLoader.load_yaml_with_extends(str(right_path))

    diffs = _collect_diffs(left_resolved, right_resolved, path="")
    diffs.sort(key=lambda entry: entry[0])

    print("Resolved config diff:")
    print(f"  left:  {_format_path(left_path)}")
    print(f"  right: {_format_path(right_path)}")
    print(f"  total_diffs: {len(diffs)}")
    print("")

    if not diffs:
        print("[OK] No differences.")
        return 0

    if profile != "parity":
        raise ValueError(f"Unsupported --profile: {profile} (supported: parity)")

    aug_disabled_both = (not _augmentation_enabled(left_resolved)) and (
        not _augmentation_enabled(right_resolved)
    )

    allowed: list[tuple[str, object, object]] = []
    forbidden: list[tuple[str, object, object]] = []
    for diff in diffs:
        if diff[0] in ALLOWED_PARITY_DIFF_PATHS:
            allowed.append(diff)
        elif aug_disabled_both and _is_augmentation_related_diff_path(diff[0]):
            allowed.append(diff)
        else:
            forbidden.append(diff)

    if allowed:
        print(f"Allowed diffs ({len(allowed)}):")
        for path, left_value, right_value in allowed:
            print(
                f"  - {path}: {_format_diff_value(left_value)} -> {_format_diff_value(right_value)}"
            )
        print("")

    if forbidden:
        print(f"Forbidden diffs ({len(forbidden)}):")
        for path, left_value, right_value in forbidden:
            print(
                f"  - {path}: {_format_diff_value(left_value)} -> {_format_diff_value(right_value)}"
            )
        print("")
        print("[FAIL] Parity profile rejected forbidden diffs.")
        return 2

    print("[OK] Parity profile passed (only allowed diffs).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect/diff YAML configs after resolving 'extends'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Print extends chain + resolved config.",
    )
    inspect_parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    inspect_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        default="json",
        help="Output format for resolved config payload (canonical: json)",
    )

    diff_parser = subparsers.add_parser(
        "diff",
        help="Diff two configs after resolving 'extends'.",
    )
    diff_parser.add_argument(
        "--left", type=Path, required=True, help="Left config YAML"
    )
    diff_parser.add_argument(
        "--right", type=Path, required=True, help="Right config YAML"
    )
    diff_parser.add_argument(
        "--profile",
        choices=("parity",),
        default="parity",
        help="Diff profile (parity gates semantic changes; exit!=0 on forbidden diffs)",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        config_path = _resolve_user_path(args.config)
        return _cmd_inspect(config_path=config_path, output_format=str(args.format))

    if args.command == "diff":
        left_path = _resolve_user_path(args.left)
        right_path = _resolve_user_path(args.right)
        return _cmd_diff(
            left_path=left_path, right_path=right_path, profile=str(args.profile)
        )

    raise AssertionError(f"Unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
