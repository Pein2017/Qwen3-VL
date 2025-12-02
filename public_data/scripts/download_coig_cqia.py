#!/usr/bin/env python3
"""
Download AI-ModelScope/COIG-CQIA via ms-swift and export JSONL files under public_data.

Features:
- Select one or more subsets (or all).
- Optional row sampling for quick smoke tests.
- Writes one JSONL per subset and an optional merged JSONL.
- Uses MS cache under public_data by default to avoid polluting global caches.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from swift.llm.dataset import load_dataset

DATASET_ID = "AI-ModelScope/COIG-CQIA"
AVAILABLE_SUBSETS: List[str] = [
    "chinese_traditional",
    "coig_pc",
    "exam",
    "finance",
    "douban",
    "human_value",
    "logi_qa",
    "ruozhiba",
    "segmentfault",
    "wiki",
    "wikihow",
    "xhs",
    "zhihu",
]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "coig_cqia"
DEFAULT_CACHE = ROOT / ".ms_cache"

# Allow importing sibling formatter without packaging public_data/.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COIG-CQIA subsets and export to JSONL under public_data."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all"],
        help="Subsets to download (default: all).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Limit rows per subset for smoke tests (0 = full dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Cache root for ModelScope/HF (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Force Hugging Face hub instead of ModelScope.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Parallel workers for preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even if output files already exist.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also write a merged JSONL of all downloaded subsets.",
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="Run formatting pass (metadata/source tag, drop message loss) on outputs.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="coig_cqia",
        help="metadata.source value to inject during formatting (default: coig_cqia).",
    )
    return parser.parse_args()


def resolve_subsets(requested: List[str]) -> List[str]:
    if "all" in requested:
        return AVAILABLE_SUBSETS
    unknown = set(requested) - set(AVAILABLE_SUBSETS)
    if unknown:
        raise SystemExit(f"Unknown subsets: {', '.join(sorted(unknown))}")
    return requested


def set_caches(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MS_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf"))


def export_subset(
    subset: str,
    *,
    sample: int,
    output_dir: Path,
    use_hf: bool,
    num_proc: int,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subset}.jsonl"

    if out_path.exists() and not overwrite:
        print(f"Skip {subset}: {out_path} exists (use --overwrite to regenerate).")
        return out_path

    dataset_spec = f"{DATASET_ID}:{subset}"
    if sample > 0:
        dataset_spec = f"{dataset_spec}#{sample}"

    print(f"Loading {dataset_spec} (use_hf={use_hf})...")
    train_ds, _ = load_dataset(
        dataset_spec,
        num_proc=num_proc,
        use_hf=use_hf,
        download_mode="reuse_dataset_if_exists",
        remove_unused_columns=False,
    )

    if sample > 0 and len(train_ds) > sample:
        train_ds = train_ds.select(range(sample))

    print(f"Writing {out_path} ({len(train_ds)} rows)...")
    train_ds.to_json(
        str(out_path),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return out_path


def format_jsonl(path: Path, *, source_name: str) -> Path:
    from format_coig_cqia import format_file

    print(f"Formatting {path}...")
    format_file(path, path, source_name=source_name)
    return path


def main() -> None:
    args = parse_args()
    subsets = resolve_subsets(args.subsets)
    set_caches(args.cache_root.resolve())

    exported = []
    for subset in subsets:
        out = export_subset(
            subset,
            sample=args.sample,
            output_dir=args.output_dir.resolve(),
            use_hf=args.use_hf,
            num_proc=args.num_proc,
            overwrite=args.overwrite,
        )
        if args.format:
            out = format_jsonl(out, source_name=args.source_name)
        exported.append(out)

    if args.merge and exported:
        merged_path = args.output_dir.resolve() / "coig_cqia_merged.jsonl"
        if merged_path.exists() and not args.overwrite:
            print(f"Skip merge: {merged_path} exists (use --overwrite to regenerate).")
        else:
            print(f"Merging {len(exported)} files -> {merged_path}")
            with merged_path.open("w", encoding="utf-8") as fout:
                for path in exported:
                    with path.open("r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
            if args.format:
                format_jsonl(merged_path, source_name=args.source_name)
    print("Done.")


if __name__ == "__main__":
    main()
