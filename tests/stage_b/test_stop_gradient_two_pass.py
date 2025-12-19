from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.stage_b.config import ReflectionConfig
from src.stage_b.reflection import ReflectionEngine
from src.stage_b.runner import (
    _PendingRuleFeedback,
    _batch_size_for_retry,
    _drain_buffered_feedback,
    _is_gradient_candidate,
)
from src.stage_b.types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceRecord,
    GroupTicket,
    MissionGuidance,
    StageASummaries,
)


class _FakeModel:
    device = "cpu"


class _FakeTokenizer:
    class _Ids:
        def __init__(self, n: int) -> None:
            self._n = n

        def size(self, dim: int) -> int:
            return self._n

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, **kwargs):
        n_tokens = max(1, len(text) // 6)
        return {"input_ids": self._Ids(n_tokens)}


class _FakeGuidanceRepo:
    def load(self):
        return {}


class _GuidanceRepoWithRule:
    def load(self):
        return {
            "m": MissionGuidance(
                mission="m",
                experiences={
                    "G0": "初始检查清单",
                    "G1": "若能确认关键点齐全则通过，否则不通过。",
                },
                step=1,
                updated_at=datetime.now(timezone.utc),
            )
        }


def _mk_bundle(*, mission: str, group_ids: tuple[str, ...]) -> ExperienceBundle:
    records = []
    for gid in group_ids:
        ticket = GroupTicket(
            group_id=gid,
            mission=mission,
            label="pass",  # type: ignore[arg-type]
            summaries=StageASummaries(per_image={"image_1": "摘要"}),
        )
        cand = ExperienceCandidate(
            candidate_index=0,
            verdict="pass",  # type: ignore[arg-type]
            reason="ok",
            signals=DeterministicSignals(label_match=True, self_consistency=None),
        )
        records.append(
            ExperienceRecord(
                ticket=ticket,
                candidates=(cand,),
                winning_candidate=0,
                guidance_step=1,
            )
        )
    return ExperienceBundle(mission=mission, records=tuple(records), reflection_cycle=0, guidance_step=1)


def test_is_gradient_candidate_contradiction_only() -> None:
    assert _is_gradient_candidate(
        label_match=True,
        low_agreement=False,
        conflict_flag=False,
        needs_manual_review=False,
        candidate_verdicts=["pass", "fail"],
    )


def test_retry_batch_size_shrinks_deterministically() -> None:
    assert _batch_size_for_retry(32, attempt=0) == 32
    assert _batch_size_for_retry(32, attempt=1) == 16
    assert _batch_size_for_retry(32, attempt=2) == 8
    assert _batch_size_for_retry(3, attempt=2) == 1


def test_drain_buffered_feedback_excludes_stop_gradient() -> None:
    pending = {
        "g_stop": _PendingRuleFeedback(experience_keys=("G1",), label_match=False),
        "g_ok": _PendingRuleFeedback(experience_keys=("G1",), label_match=True),
    }
    to_commit, to_drop = _drain_buffered_feedback(
        pending,
        stop_gradient_ticket_keys={"g_stop"},
        contributor_ticket_keys={"g_ok"},
    )
    assert len(to_commit) == 1 and to_commit[0].label_match is True
    assert len(to_drop) == 1 and to_drop[0].label_match is False
    assert pending == {}


def test_ops_pass_rejects_evidence_outside_learnable(tmp_path: Path) -> None:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=2,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    engine = ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )

    bundle = _mk_bundle(mission="m", group_ids=("g1", "g2"))

    # Evidence includes an out-of-bundle group_id -> op must be rejected.
    engine._generate_json_payload = lambda **kwargs: {  # type: ignore[assignment]
        "operations": [
            {
                "op": "add",
                "text": "若能确认关键点齐全则通过，否则不通过。",
                "rationale": "test",
                "evidence": ["g1::pass", "STOP"],
            }
        ]
    }

    operations, hypotheses, evidence_group_ids, _analysis = engine.run_ops_pass(bundle)
    assert operations == tuple()
    assert hypotheses == tuple()
    assert evidence_group_ids == tuple()


def test_ops_pass_keeps_only_valid_evidence(tmp_path: Path) -> None:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=2,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    engine = ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )

    bundle = _mk_bundle(mission="m", group_ids=("g1", "g2"))

    engine._generate_json_payload = lambda **kwargs: {  # type: ignore[assignment]
        "operations": [
            {
                "op": "add",
                "text": "若能确认关键点齐全则通过，否则不通过。",
                "rationale": "valid",
                "evidence": ["g2::pass"],
            },
            {
                "op": "add",
                "text": "无效 op（证据越界）。",
                "rationale": "invalid",
                "evidence": ["STOP"],
            },
        ]
    }

    operations, hypotheses, evidence_group_ids, _analysis = engine.run_ops_pass(bundle)
    assert len(operations) == 1
    assert hypotheses == tuple()
    assert evidence_group_ids == ("g2::pass",)


def test_ops_pass_filters_third_state_text(tmp_path: Path) -> None:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=1,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    engine = ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )

    bundle = _mk_bundle(mission="m", group_ids=("g1",))

    engine._generate_json_payload = lambda **kwargs: {  # type: ignore[assignment]
        "operations": [
            {
                "op": "add",
                "text": "需复核时通过。",
                "rationale": "test",
                "evidence": ["g1::pass"],
            }
        ]
    }

    operations, hypotheses, evidence_group_ids, _analysis = engine.run_ops_pass(bundle)
    assert operations == tuple()
    assert hypotheses == tuple()
    assert evidence_group_ids == tuple()


def test_ops_pass_prefers_update_for_similar_add(tmp_path: Path) -> None:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=1,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    engine = ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_GuidanceRepoWithRule(),  # type: ignore[arg-type]
    )
    bundle = _mk_bundle(mission="m", group_ids=("g1",))
    engine._generate_json_payload = lambda **kwargs: {  # type: ignore[assignment]
        "operations": [
            {
                "op": "add",
                "text": "若能确认关键点齐全则通过，否则不通过。",
                "rationale": "similar",
                "evidence": ["g1::pass"],
            }
        ]
    }
    operations, _hypotheses, _evidence_group_ids, _analysis = engine.run_ops_pass(bundle)
    assert len(operations) == 1
    assert operations[0].key == "G1"


def test_decision_pass_resolves_shorthand_ticket_ids(tmp_path: Path) -> None:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=2,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    engine = ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )

    bundle = _mk_bundle(mission="m", group_ids=("g1", "g2"))

    # "第1组" must resolve to the first ticket_key, and unknown ids must be ignored.
    engine._generate_json_payload = lambda **kwargs: {  # type: ignore[assignment]
        "no_evidence_group_ids": ["第1组", "g2::pass", "UNKNOWN"],
        "decision_analysis": "ok",
    }

    no_evidence, analysis = engine.run_decision_pass(bundle)
    assert set(no_evidence) == {"g1::pass", "g2::pass"}
    assert analysis == "ok"
