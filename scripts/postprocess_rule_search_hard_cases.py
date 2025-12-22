#!/usr/bin/env python3
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
    df = df[df["pass_flag"] == 2]
    df = df.dropna(subset=["group_id", "mission", "reason_text"])

    mapping: dict[tuple[str, str], set[str]] = {}
    for group_id, mission, reason in df[["group_id", "mission", "reason_text"]].itertuples(
        index=False
    ):
        gid = str(group_id).strip()
        mis = str(mission).strip()
        reason_text = str(reason).strip()
        if not gid or not mis or not reason_text or reason_text.lower() == "nan":
            continue
        mapping.setdefault((gid, mis), set()).add(reason_text)

    return {k: sorted(v) for k, v in mapping.items()}


def _iter_hard_case_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/rule_search_hard_cases.jsonl" if recursive else "*/rule_search_hard_cases.jsonl"
    return sorted(input_path.glob(pattern))


def _default_output_path(input_file: Path) -> Path:
    return input_file.with_name(input_file.name.replace("hard_cases", "hard_samples"))


def _process_file(
    input_file: Path,
    output_file: Path,
    reason_map: dict[tuple[str, str], list[str]],
    overwrite: bool,
) -> dict[str, int]:
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_file}")

    total = 0
    selected = 0
    missing_reason = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            gt_label = _normalize_label(obj.get("gt_label"))
            pred_label = _normalize_label(obj.get("majority_pred"))
            if not (gt_label == "fail" and pred_label == "pass"):
                continue

            ticket_key = str(obj.get("ticket_key", "")).strip()
            group_id = ticket_key.split("::")[0] if "::" in ticket_key else ticket_key
            mission = str(obj.get("mission", "")).strip()
            reasons = reason_map.get((group_id, mission), [])
            if not reasons:
                missing_reason += 1

            obj_with_reason = {}
            for key, value in obj.items():
                obj_with_reason[key] = value
                if key == "gt_label":
                    obj_with_reason["gt_fail_reason_texts"] = reasons
                    obj_with_reason["gt_fail_reason_text"] = (
                        " | ".join(reasons) if reasons else None
                    )
            if "gt_label" not in obj_with_reason:
                obj_with_reason["gt_fail_reason_texts"] = reasons
                obj_with_reason["gt_fail_reason_text"] = (
                    " | ".join(reasons) if reasons else None
                )
            fout.write(json.dumps(obj_with_reason, ensure_ascii=False) + "\n")
            selected += 1

    return {"total": total, "selected": selected, "missing_reason": missing_reason}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract gt=fail & pred=pass hard cases and attach GT fail reason text."
    )
    parser.add_argument(
        "--hard-cases",
        required=True,
        help="Path to rule_search_hard_cases.jsonl or a directory containing it",
    )
    parser.add_argument(
        "--excel",
        default="output_post/BBU_scene_latest.xlsx",
        help="Excel file containing GT fail reasons",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (when input is a file) or output directory (when input is a dir)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for rule_search_hard_cases.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    args = parser.parse_args()

    input_path = Path(args.hard_cases)
    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    reason_map = _load_fail_reasons(excel_path)
    hard_case_files = _iter_hard_case_files(input_path, args.recursive)
    if not hard_case_files:
        raise FileNotFoundError(f"No rule_search_hard_cases.jsonl found under {input_path}")

    output_root = Path(args.output) if args.output else None
    summary = []
    for input_file in hard_case_files:
        if output_root is None:
            output_file = _default_output_path(input_file)
        else:
            if input_path.is_file():
                output_file = output_root
            else:
                rel = input_file.parent.relative_to(input_path)
                output_dir = output_root / rel
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / _default_output_path(input_file).name

        stats = _process_file(input_file, output_file, reason_map, args.overwrite)
        stats["input"] = str(input_file)
        stats["output"] = str(output_file)
        summary.append(stats)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
