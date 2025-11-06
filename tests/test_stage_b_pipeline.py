import json
from datetime import datetime, timezone
from typing import Any, Dict

from src.config.missions import STAGE_B_MISSION_FOCUS
from src.stage_b.io import GuidanceRepository, serialize_selection, serialize_trajectory
from src.stage_b.ingest import ingest_stage_a
from src.stage_b.judge import DeterministicJudge
from src.stage_b.prompts import build_messages
from src.stage_b.selection import SelectionConfig, select_for_group
from src.stage_b.types import (
    DecodeConfig,
    GroupLabel,
    GroupTicket,
    ParsedTrajectory,
    StageASummaries,
    Trajectory,
)


def _write_stage_a(tmp_path, records):
    path = tmp_path / "stage_a.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return path


def _write_guidance(tmp_path):
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {}
    for mission, focus in STAGE_B_MISSION_FOCUS.items():
        payload[mission] = {
            "focus": focus,
            "version": 1,
            "updated_at": now_iso,
            "guidance": [
                {"type": "rule", "text": "若挡风板缺失则判定不通过"},
            ],
            "preferences": [
                {
                    "case": "summary_low_confidence",
                    "text": "摘要置信度低时请返回不通过并说明原因",
                },
                {
                    "case": "label_contradiction",
                    "text": "如结论与人工标签冲突，请标记需要人工复核",
                },
            ],
        }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return GuidanceRepository(guidance_path, retention=1)


def test_ingest_assigns_summary_confidence(tmp_path):
    guidance_repo = _write_guidance(tmp_path)
    record = {
        "group_id": "QC-001",
        "mission": "挡风板安装检查",
        "label": "fail",
        "per_image": {"图片_1": "无关图片"},
        "label_source": "human",
        "label_timestamp": "2024-01-01T00:00:00",
    }
    stage_a_path = _write_stage_a(tmp_path, [record])

    tickets = ingest_stage_a([stage_a_path], guidance_repo)

    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.summary_confidence == "needs_review"
    assert ticket.baseline_verdict in {"通过", "不通过"}


def test_judge_scores_and_selection(tmp_path):
    guidance_repo = _write_guidance(tmp_path)
    record = {
        "group_id": "QC-002",
        "mission": "挡风板安装检查",
        "label": "fail",
        "per_image": {
            "图片_1": "BBU设备按要求配备挡风板，螺丝符合要求，备注：挡风板缺失",
        },
        "label_source": "human",
        "label_timestamp": "2024-01-01T00:00:00",
    }
    stage_a_path = _write_stage_a(tmp_path, [record])
    tickets = ingest_stage_a([stage_a_path], guidance_repo)
    ticket = tickets[0]
    guidance = guidance_repo.load()[ticket.mission]

    trajectory = ParsedTrajectory(
        base=Trajectory(
            group_id=ticket.group_id,
            mission=ticket.mission,
            candidate_index=0,
            decode=DecodeConfig(temperature=0.2, top_p=0.9, max_new_tokens=60),
            response_text="不通过\n理由: 摘要显示挡风板缺失，需要复核",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="不通过",
        reason="摘要显示挡风板缺失，需要复核",
        format_ok=True,
    )

    judge = DeterministicJudge()
    judged = judge.evaluate(ticket, guidance, trajectory)
    assert judged.scores.label_match is True
    assert judged.scores.semantic_advantage != 0

    selection_config = SelectionConfig(
        policy="top_semantic",
        tie_break="confidence",
    )
    selection = select_for_group(
        [judged],
        guidance_step=guidance.step,
        reflection_cycle=0,
        reflection_change=None,
        config=selection_config,
    )
    assert selection.group_id == ticket.group_id
    assert selection.verdict == "不通过"


def test_serialization_payloads(tmp_path):
    guidance_repo = _write_guidance(tmp_path)
    record = {
        "group_id": "QC-003",
        "mission": "挡风板安装检查",
        "label": "fail",
        "per_image": {
            "图片_1": "挡风板缺失，需要复核",
        },
    }
    stage_a_path = _write_stage_a(tmp_path, [record])
    tickets = ingest_stage_a([stage_a_path], guidance_repo)
    ticket = tickets[0]
    guidance = guidance_repo.load()[ticket.mission]

    trajectory = ParsedTrajectory(
        base=Trajectory(
            group_id=ticket.group_id,
            mission=ticket.mission,
            candidate_index=0,
            decode=DecodeConfig(
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=128,
                seed=123,
            ),
            response_text="不通过\n理由: 挡风板缺失",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="不通过",
        reason="挡风板缺失",
        format_ok=True,
    )

    judge = DeterministicJudge()
    judged = judge.evaluate(ticket, guidance, trajectory)

    traj_payload = serialize_trajectory(judged)
    assert set(traj_payload) == {"group_id", "mission", "result"}
    result_obj = traj_payload["result"]
    assert isinstance(result_obj, dict)
    result: Dict[str, Any] = result_obj
    assert result["candidate_index"] == 0
    assert result["temperature"] == 0.6
    assert result["text"].startswith("不通过")
    assert "scores" in result
    scores_obj = result["scores"]
    assert isinstance(scores_obj, dict)
    scores: Dict[str, Any] = scores_obj
    assert scores["summary_confidence"] == ticket.summary_confidence

    selection = select_for_group(
        [judged],
        guidance_step=guidance.step,
        reflection_cycle=0,
        reflection_change=None,
        config=SelectionConfig(
            policy="top_semantic",
            tie_break="confidence",
        ),
    )
    selection_payload = serialize_selection(selection)
    assert set(selection_payload) == {"group_id", "mission", "result"}
    selection_result = selection_payload["result"]
    assert isinstance(selection_result, dict)
    assert selection_result["verdict"] == "不通过"


def test_prompt_includes_preferences(tmp_path):
    guidance_repo = _write_guidance(tmp_path)
    guidance = guidance_repo.load()["挡风板安装检查"]

    ticket = GroupTicket(
        group_id="QC-TEST",
        mission="挡风板安装检查",
        label=GroupLabel("fail"),
        summaries=StageASummaries(per_image={"image_1": "无关图片"}),
    )
    messages = build_messages(ticket, guidance)
    assert messages[0]["role"] == "system"
    assert "摘要可信度" in messages[0]["content"] or "摘要" in messages[0]["content"]
