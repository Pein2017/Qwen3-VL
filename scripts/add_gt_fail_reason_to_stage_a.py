#!/usr/bin/env python3
"""Add gt_fail_reason_text fields to Stage-A JSONL output from Excel file.

This script reads Stage-A JSONL output and adds gt_fail_reason_text/gt_fail_reason_texts
fields based on Excel file containing manual audit remarks. Only records with label=fail
will have these fields added.

Note: Currently only supports BBU dataset. RRU support is not yet available.
"""

import argparse
import json
import sys
from pathlib import Path


def _maybe_add_local_site_packages() -> None:
    local_site = Path(__file__).resolve().parents[1] / ".venv"
    if local_site.exists():
        sys.path.insert(0, str(local_site))


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in {"pass", "通过", "1"}:
        return "pass"
    if s in {"fail", "不通过", "2"}:
        return "fail"
    return s


def _load_fail_reasons(excel_path: Path) -> dict[tuple[str, str], list[str]]:
    """Load fail reasons from Excel file.

    Returns:
        Mapping from (group_id, mission) to list of reason texts.
    """
    _maybe_add_local_site_packages()
    import pandas as pd

    df = pd.read_excel(
        excel_path,
        header=None,
        usecols=[0, 2, 5, 6],
        engine="openpyxl",
    )
    df.columns = ["group_id", "mission", "reason_text", "pass_flag"]
    df["pass_flag"] = pd.to_numeric(df["pass_flag"], errors="coerce")
    df = df[df["pass_flag"] == 2]  # Only fail cases (pass_flag == 2)
    df = df.dropna(subset=["group_id", "mission", "reason_text"])

    mapping: dict[tuple[str, str], set[str]] = {}
    for group_id, mission, reason in df[
        ["group_id", "mission", "reason_text"]
    ].itertuples(index=False):
        gid = str(group_id).strip()
        mis = str(mission).strip()
        reason_text = str(reason).strip()
        if not gid or not mis or not reason_text or reason_text.lower() == "nan":
            continue
        mapping.setdefault((gid, mis), set()).add(reason_text)

    return {k: sorted(v) for k, v in mapping.items()}


def _process_stage_a_jsonl(
    input_file: Path,
    reason_map: dict[tuple[str, str], list[str]],
    inplace: bool = False,
) -> dict[str, int]:
    """Process Stage-A JSONL file and add gt_fail_reason_text fields.

    Args:
        input_file: Path to Stage-A JSONL file
        reason_map: Mapping from (group_id, mission) to list of reason texts
        inplace: If True, overwrite input file; otherwise, write to stdout

    Returns:
        Statistics dict with counts
    """
    total = 0
    updated = 0
    missing_reason = 0

    tmp_path: Path | None = None
    fout = None
    try:
        if inplace:
            tmp_path = input_file.with_suffix(input_file.suffix + ".tmp")
            fout = tmp_path.open("w", encoding="utf-8")

        with input_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                obj = json.loads(line)

                # Only add gt_fail_reason_text for fail-labeled records
                label = _normalize_label(obj.get("label"))
                if label == "fail":
                    group_id = str(obj.get("group_id", "")).strip()
                    mission = str(obj.get("mission", "")).strip()
                    reasons = reason_map.get((group_id, mission), [])

                    if reasons:
                        obj["gt_fail_reason_texts"] = reasons
                        obj["gt_fail_reason_text"] = " | ".join(reasons)
                        updated += 1
                    else:
                        # Still add fields but with None/empty list for records without reasons
                        obj["gt_fail_reason_texts"] = []
                        obj["gt_fail_reason_text"] = None
                        missing_reason += 1

                if inplace:
                    assert fout is not None
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    print(json.dumps(obj, ensure_ascii=False))
    finally:
        if fout is not None:
            fout.close()

    if inplace:
        assert tmp_path is not None
        tmp_path.replace(input_file)

    return {
        "total": total,
        "updated": updated,
        "missing_reason": missing_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add gt_fail_reason_text fields to Stage-A JSONL output from Excel file."
    )
    parser.add_argument(
        "input_jsonl",
        type=Path,
        help="Path to Stage-A JSONL file",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("output_post/BBU_scene_latest.xlsx"),
        help="Excel file containing GT fail reasons (default: output_post/BBU_scene_latest.xlsx)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file in place (default: write to stdout)",
    )
    args = parser.parse_args()

    input_file = args.input_jsonl
    excel_path = args.excel

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}", file=sys.stderr)
        return 1

    if not excel_path.exists():
        print(f"[WARNING] Excel file not found: {excel_path}", file=sys.stderr)
        print("[WARNING] Continuing without fail reasons...", file=sys.stderr)
        reason_map = {}
    else:
        reason_map = _load_fail_reasons(excel_path)

    stats = _process_stage_a_jsonl(input_file, reason_map, inplace=args.inplace)

    if args.inplace:
        print(
            json.dumps(
                {
                    "input": str(input_file),
                    "excel": str(excel_path),
                    **stats,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
