#!/usr/bin/env python3
"""
Stage-B smoke / audit script (no model required).

Goals:
- Validate Stage-B config parsing (YAML -> StageBConfig).
- Validate Stage-A ingest contract (JSONL -> GroupTicket).
- Validate mission guidance load (guidance.json -> MissionGuidance).
- Validate prompt building (system+user message construction).
- Validate strict 2-line output parsing.

Usage:
  conda run -n ms python scripts/stage_b_smoke.py
  conda run -n ms python scripts/stage_b_smoke.py --config configs/stage_b/bbu_line.yaml
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from src.stage_b.config import load_stage_b_config
from src.stage_b.ingest.stage_a import ingest_stage_a
from src.stage_b.io.guidance import GuidanceRepository
from src.stage_b.rollout import _parse_two_line_response
from src.stage_b.sampling.prompts import build_messages
from src.stage_b.types import GroupTicket, MissionGuidance, StageASummaries


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _workspace_tmp_root() -> Path:
    repo_dir = Path(__file__).resolve().parent.parent
    tmp_root = repo_dir / "output" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-B smoke/audit (no model)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional existing Stage-B YAML config to validate (will ingest paths).",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: Tuple[Dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _write_guidance_json(path: Path, *, mission: str) -> None:
    payload = {
        mission: {
            "step": 0,
            "updated_at": _now().isoformat(),
            "experiences": {
                "G0": "只围绕当前 mission 的关键检查项做结论。",
                "G1": "出现明确负项（缺失/错误/松动/不符合要求）且与G0相关 → 判不通过。",
            },
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_temp_config(tmpdir: Path) -> Path:
    stage_a_path = tmpdir / "stage_a.jsonl"
    guidance_path = tmpdir / "guidance.json"
    out_root = tmpdir / "out"
    cfg_path = tmpdir / "stage_b_smoke.yaml"

    mission = "SMOKE_MISSION"
    _write_jsonl(
        stage_a_path,
        (
            {
                "mission": mission,
                "group_id": "group_1",
                "label": "pass",
                "per_image": {
                    "image_1": "设备齐全，标签清晰。",
                    "image_2": "未见明显缺失或错误。",
                },
                "label_source": "smoke",
                "label_timestamp": _now().isoformat(),
            },
        ),
    )
    _write_guidance_json(guidance_path, mission=mission)

    cfg_text = f"""
stage_a_paths:
  - {stage_a_path.as_posix()}
seed: 17
default_domain: bbu

model:
  model_name_or_path: dummy
  torch_dtype: bfloat16
  device_map: auto

guidance:
  path: {guidance_path.as_posix()}
  retention: 3
  reset_on_rerun: false

output:
  root: {out_root.as_posix()}
  run_name: smoke

rule_search:
  proposer_prompt_path: configs/prompts/stage_b_rule_search_proposer_prompt.txt
  reflect_size: 1
  num_candidate_rules: 1

  train_pool_size: 4
  eval_pool_size: 2

  gate:
    min_relative_error_reduction: 0.0
    max_changed_fraction: 1.0
    max_fp_rate_increase: 1.0
    bootstrap:
      iterations: 10
      min_prob: 0.0
      seed: 17

  early_stop:
    patience: 1

  train_sampler:
    samples_per_decode: 1
    grid:
      - temperature: 0.2
        top_p: 0.9
        max_new_tokens: 64
        seed: 42
        repetition_penalty: 1.05
        stop: ["assistant", "<|im_end|>", "<|endoftext|>", "</s>"]
  eval_sampler:
    samples_per_decode: 1
    grid:
      - temperature: 0.2
        top_p: 0.9
        max_new_tokens: 64
        seed: 42
        repetition_penalty: 1.05
        stop: ["assistant", "<|im_end|>", "<|endoftext|>", "</s>"]

reflection:
  decision_prompt_path: configs/prompts/stage_b_reflection_decision_prompt.txt
  ops_prompt_path: configs/prompts/stage_b_reflection_ops_prompt.txt
  batch_size: 1
  max_operations: 1
  temperature: 0.001
  top_p: 0.9
  max_new_tokens: 512
  max_reflection_length: 2048
  token_budget: 2048

runner:
  epochs: 1
  rollout_batch_size: 1

stage_b_distillation:
  enabled: false
""".lstrip()

    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def _ensure_messages(messages: Any) -> None:
    if not isinstance(messages, list) or len(messages) != 2:
        raise AssertionError("build_messages must return 2 turns: system + user")
    if messages[0].get("role") != "system":
        raise AssertionError("First message must be system")
    if messages[1].get("role") != "user":
        raise AssertionError("Second message must be user")
    for msg in messages:
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise AssertionError("Messages must contain non-empty string content")


def _build_minimal_ticket(mission: str) -> GroupTicket:
    return GroupTicket(
        group_id="group_1",
        mission=mission,
        label="pass",
        summaries=StageASummaries(per_image={"image_1": "设备齐全，标签清晰。"}),
        provenance=None,
    )


def _build_minimal_guidance(mission: str) -> MissionGuidance:
    return MissionGuidance(
        mission=mission,
        experiences={"G0": "只围绕当前 mission 的关键检查项做结论。", "G1": "发现明确负项 → 判不通过。"},
        step=0,
        updated_at=_now(),
        metadata={},
    )


def _audit_output_parser() -> Tuple[str, str]:
    ok, verdict, reason = _parse_two_line_response("Verdict: 通过\nReason: Image1: 正常; 总结: 通过")
    if not ok or verdict != "pass" or not reason:
        raise AssertionError("Two-line parser failed on a valid response")

    bad_ok, _, _ = _parse_two_line_response("Verdict: 通过\nReason: 通过但需复核")
    if bad_ok:
        raise AssertionError("Two-line parser should reject forbidden third-state terms")

    return verdict, reason


def main() -> None:
    args = _build_args()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        config = load_stage_b_config(cfg_path)
        tickets = ingest_stage_a(config.stage_a_paths)
        repo = GuidanceRepository(config.guidance.path, retention=config.guidance.retention)
        guidance_map = repo.load()
        ticket0 = tickets[0]
        guidance0 = guidance_map.get(ticket0.mission)
        if guidance0 is None:
            raise RuntimeError(f"Guidance missing for mission: {ticket0.mission}")
        messages = build_messages(ticket0, guidance0)
        _ensure_messages(messages)
        _audit_output_parser()
        print("[OK] Stage-B config+ingest+guidance+prompt+parse smoke passed.")
        return

    with tempfile.TemporaryDirectory(
        prefix="stage_b_smoke_", dir=str(_workspace_tmp_root())
    ) as tmp:
        tmpdir = Path(tmp)
        cfg_path = _make_temp_config(tmpdir)
        config = load_stage_b_config(cfg_path)

        # Ingest Stage-A sample
        tickets = ingest_stage_a(config.stage_a_paths)
        ticket0 = tickets[0]

        # Guidance load
        repo = GuidanceRepository(config.guidance.path, retention=config.guidance.retention)
        guidance_map = repo.load()
        guidance0 = guidance_map.get(ticket0.mission)
        if guidance0 is None:
            raise RuntimeError("Generated guidance file did not load mission section")

        # Prompt build
        messages = build_messages(ticket0, guidance0)
        _ensure_messages(messages)

        # Output parsing + export helpers
        _audit_output_parser()

        # Also verify prompt building works without file IO.
        _ensure_messages(build_messages(_build_minimal_ticket(ticket0.mission), _build_minimal_guidance(ticket0.mission)))

        print("[OK] Stage-B smoke passed (no model).")


if __name__ == "__main__":
    main()
