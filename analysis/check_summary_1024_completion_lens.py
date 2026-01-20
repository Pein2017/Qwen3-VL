"""Check summary completion token lengths for 1024 summary datasets.

Constraints:
- Load *tokenizer only* (no model weights).
- Stream JSONL line-by-line (no full-file RAM load).

Usage:
  conda run -n ms python analysis/check_summary_1024_completion_lens.py

Optional:
  conda run -n ms python analysis/check_summary_1024_completion_lens.py \
    --config configs/train/grpo/summary_1024.yaml \
    --limit 2048 \
    --with-special
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries; override wins.

    Lists are replaced, not concatenated.
    """

    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return obj


def load_config_with_extends(path: Path) -> Dict[str, Any]:
    """Load YAML config, resolving  recursively.

    Convention: child overrides parent.
    """

    raw = _load_yaml(path)
    extends = raw.get("extends")
    if not extends:
        return raw

    if isinstance(extends, (str, Path)):
        extends_list = [extends]
    elif isinstance(extends, list):
        extends_list = extends
    else:
        raise TypeError(f"Invalid extends type in {path}: {type(extends)}")

    merged: Dict[str, Any] = {}
    for rel in extends_list:
        parent_path = (path.parent / str(rel)).resolve()
        parent_cfg = load_config_with_extends(parent_path)
        merged = _deep_merge(merged, parent_cfg)

    # Child overrides parent
    raw_wo_extends = dict(raw)
    raw_wo_extends.pop("extends", None)
    merged = _deep_merge(merged, raw_wo_extends)
    return merged


def _pick_first(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def find_tokenizer_path(
    cfg: Mapping[str, Any], config_path: Path
) -> Tuple[str, Dict[str, Any]]:
    """Best-effort to identify which tokenizer path training uses.

    Returns (tokenizer_path, debug_info).
    """

    debug: Dict[str, Any] = {}

    # Common key patterns across trainer stacks.
    direct = _pick_first(
        cfg,
        [
            "tokenizer_name_or_path",
            "tokenizer_path",
            "tokenizer",
        ],
    )
    if isinstance(direct, str) and direct.strip():
        debug["picked"] = "root.tokenizer*"
        return direct, debug

    model_cfg = cfg.get("model")
    if isinstance(model_cfg, Mapping):
        direct2 = _pick_first(
            model_cfg,
            [
                "tokenizer_name_or_path",
                "tokenizer_path",
                "tokenizer",
                "model_name_or_path",
                "model",
                "model_id",
            ],
        )
        if isinstance(direct2, str) and direct2.strip():
            # In this repo the config uses .
            debug["picked"] = "model.*"
            return direct2, debug

    # Fallback: from known checkpoint in repo.
    ckpt = Path(
        "output/1-13/new_schema-4B-dense-grpo_summary_1024_attr_key_recall/ckpt-9900"
    )
    if ckpt.exists():
        debug["picked"] = "fallback.output/1-13/.../ckpt-9900"
        return str(ckpt), debug

    raise FileNotFoundError(
        "Could not find tokenizer path in config, and fallback checkpoint does not exist. "
        f"Config was: {config_path}"
    )


def _stringify_completion(value: Any) -> str:
    """Convert potentially structured completion into a text string.

    The project sometimes stores JSON as a string; in that case keep it as-is.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # If the completion is structured (dict/list), serialize compactly.
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # Last resort
    return str(value)


def extract_summary_completion(record: Mapping[str, Any]) -> str:
    """Extract the summary ground-truth completion text.

    Per the dataset samples in data_new_schema_center/*_full_1024/all_samples.jsonl,
    the completion lives in .
    """

    if "summary" in record:
        return _stringify_completion(record.get("summary"))

    # If schema ever changes, keep a tiny fallback list.
    for k in ("completion", "output", "answer", "response"):
        if k in record:
            return _stringify_completion(record.get(k))

    # Messages-style fallback: last assistant message.
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return _stringify_completion(m.get("content"))

    return ""


def sample_id_from_record(record: Mapping[str, Any]) -> Optional[str]:
    for k in ("id", "sample_id", "uid", "uuid"):
        v = record.get(k)
        if isinstance(v, (str, int)):
            return str(v)

    imgs = record.get("images")
    if isinstance(imgs, list) and imgs:
        first = imgs[0]
        if isinstance(first, str):
            return first

    return None


@dataclass(frozen=True)
class SampleStat:
    length_no_special: int
    length_with_special: int
    dataset: str
    line_idx: int
    sample_id: Optional[str]


@dataclass
class AggregateStats:
    dataset: str
    total_samples: int = 0
    empty_completions: int = 0
    max_no_special: Optional[SampleStat] = None
    max_with_special: Optional[SampleStat] = None
    num_exceed_limit_no_special: int = 0
    worst_overage_no_special: int = 0
    num_exceed_limit_with_special: int = 0
    worst_overage_with_special: int = 0
    # keep top-K by no-special length
    # Use a deterministic tie-breaker so heap ops never compare SampleStat.
    topk_heap: List[Tuple[int, int, SampleStat]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.topk_heap is None:
            self.topk_heap = []


def update_stats(stats: AggregateStats, s: SampleStat, k: int, limit: int) -> None:
    stats.total_samples += 1

    if s.length_no_special == 0:
        stats.empty_completions += 1

    if (
        stats.max_no_special is None
        or s.length_no_special > stats.max_no_special.length_no_special
    ):
        stats.max_no_special = s

    if (
        stats.max_with_special is None
        or s.length_with_special > stats.max_with_special.length_with_special
    ):
        stats.max_with_special = s

    if s.length_no_special > limit:
        stats.num_exceed_limit_no_special += 1
        stats.worst_overage_no_special = max(
            stats.worst_overage_no_special, s.length_no_special - limit
        )

    if s.length_with_special > limit:
        stats.num_exceed_limit_with_special += 1
        stats.worst_overage_with_special = max(
            stats.worst_overage_with_special, s.length_with_special - limit
        )

    # top-K by no-special
    if len(stats.topk_heap) < k:
        heapq.heappush(stats.topk_heap, (s.length_no_special, s.line_idx, s))
    else:
        if s.length_no_special > stats.topk_heap[0][0]:
            heapq.heapreplace(stats.topk_heap, (s.length_no_special, s.line_idx, s))


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/train/grpo/summary_1024.yaml",
        help="Training config used to identify tokenizer path.",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer path/name_or_path (otherwise inferred from config).",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data_new_schema_center/bbu_full_1024/all_samples.jsonl",
            "data_new_schema_center/rru_full_1024/all_samples.jsonl",
        ],
    )
    ap.add_argument(
        "--limit", type=int, default=2048, help="max_completion_length limit"
    )
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument(
        "--with-special",
        action="store_true",
        help="Also compute add_special_tokens=True lengths (slower).",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Debug: stop after N non-empty lines per dataset.",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_config_with_extends(config_path)

    if args.tokenizer:
        tokenizer_path = args.tokenizer
        tok_debug: Dict[str, Any] = {"picked": "--tokenizer override"}
    else:
        tokenizer_path, tok_debug = find_tokenizer_path(cfg, config_path)

    print("Tokenizer discovery")
    print("- config:", str(config_path))
    print("- tokenizer_path:", tokenizer_path)
    print("- debug:", tok_debug)

    from transformers import AutoTokenizer  # type: ignore

    # Important: tokenizer only; does not load model weights.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("- tokenizer_class:", tokenizer.__class__.__name__)
    print("- vocab_size:", getattr(tokenizer, "vocab_size", None))

    per_dataset: List[AggregateStats] = []
    combined = AggregateStats(dataset="COMBINED")

    for ds in args.datasets:
        ds_path = Path(ds)
        if not ds_path.exists():
            raise FileNotFoundError(ds)

        stats = AggregateStats(dataset=ds)
        per_dataset.append(stats)

        n_seen = 0
        for line_idx, rec in iter_jsonl(ds_path):
            completion = extract_summary_completion(rec).strip()

            # Primary metric (no special tokens)
            len_no_special = len(tokenizer.encode(completion, add_special_tokens=False))

            # Optional conservative metric (with special tokens)
            if args.with_special:
                len_with_special = len(
                    tokenizer.encode(completion, add_special_tokens=True)
                )
            else:
                len_with_special = len_no_special

            s = SampleStat(
                length_no_special=len_no_special,
                length_with_special=len_with_special,
                dataset=ds,
                line_idx=line_idx,
                sample_id=sample_id_from_record(rec),
            )

            update_stats(stats, s, k=args.topk, limit=args.limit)
            update_stats(combined, s, k=args.topk, limit=args.limit)

            n_seen += 1
            if args.max_lines is not None and n_seen >= args.max_lines:
                break

    def _fmt_sample(s: Optional[SampleStat]) -> str:
        if s is None:
            return "(none)"
        sid = s.sample_id or "(no-id)"
        return (
            f"len_no_special={s.length_no_special} "
            f"len_with_special={s.length_with_special} "
            f"line_idx={s.line_idx} id={sid}"
        )

    def _print_report(stats: AggregateStats) -> None:
        print("\n===", stats.dataset)
        print("total_samples:", stats.total_samples)
        print("empty_completions:", stats.empty_completions)
        if stats.max_no_special is not None:
            print("max_len_no_special:", stats.max_no_special.length_no_special)
            print("max_len_no_special_sample:", _fmt_sample(stats.max_no_special))
        if stats.max_with_special is not None:
            print("max_len_with_special:", stats.max_with_special.length_with_special)
            print("max_len_with_special_sample:", _fmt_sample(stats.max_with_special))

        print(
            f"count_exceed_{args.limit}_no_special:", stats.num_exceed_limit_no_special
        )
        print("worst_overage_no_special:", stats.worst_overage_no_special)
        if args.with_special:
            print(
                f"count_exceed_{args.limit}_with_special:",
                stats.num_exceed_limit_with_special,
            )
            print("worst_overage_with_special:", stats.worst_overage_with_special)

        top_sorted = sorted(
            (s for _, __, s in stats.topk_heap),
            key=lambda x: x.length_no_special,
            reverse=True,
        )
        print(f"top_{args.topk}_longest_by_no_special:")
        for rank, s in enumerate(top_sorted, 1):
            sid = s.sample_id or "(no-id)"
            print(
                f"{rank:>2}. len_no_special={s.length_no_special} "
                f"len_with_special={s.length_with_special} "
                f"line_idx={s.line_idx} id={sid}"
            )

    for st in per_dataset:
        _print_report(st)
    _print_report(combined)

    # Final conclusion on no-special metric.
    worst = combined.max_no_special.length_no_special if combined.max_no_special else 0
    margin = args.limit - worst
    print("\nConclusion")
    if worst <= args.limit:
        print(
            f"SAFE for max_completion_length={args.limit} (max={worst}, margin={margin})"
        )
    else:
        print(
            f"NOT SAFE for max_completion_length={args.limit} (max={worst}, over_by={-margin})"
        )
        print(f"Minimal suggested max_completion_length: {worst + 16}")
        print(
            "Truncation policy option: hard truncate completion to limit tokens (right-truncate)."
        )


if __name__ == "__main__":
    main()
