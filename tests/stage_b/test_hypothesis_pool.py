from __future__ import annotations

from src.stage_b.io.hypotheses import HypothesisPool
from src.stage_b.runner import _compute_learnability_coverage
from src.stage_b.types import HypothesisCandidate


def test_hypothesis_promotion_threshold(tmp_path):
    pool = HypothesisPool(
        pool_path=tmp_path / "hypotheses.json",
        events_path=tmp_path / "hypothesis_events.jsonl",
        min_support_cycles=2,
        min_unique_ticket_keys=2,
    )
    hyp1 = HypothesisCandidate(
        text="若能确认关键点齐全则通过，否则不通过。",
        falsifier="若关键点缺失则不通过。",
        evidence=("g1::pass",),
    )
    eligible = pool.record_proposals(
        [hyp1], reflection_cycle=0, epoch=1, allow_promote=True
    )
    assert eligible == tuple()

    hyp2 = HypothesisCandidate(
        text="若能确认关键点齐全则通过，否则不通过。",
        falsifier="若关键点缺失则不通过。",
        evidence=("g2::pass",),
    )
    eligible = pool.record_proposals(
        [hyp2], reflection_cycle=1, epoch=1, allow_promote=True
    )
    assert len(eligible) == 1

    promoted = pool.mark_promoted(
        [eligible[0].signature], reflection_cycle=1, epoch=1
    )
    assert promoted and promoted[0].status == "promoted"

    reloaded = pool.load()
    record = reloaded[eligible[0].signature]
    assert record.status == "promoted"
    assert set(record.support_ticket_keys) == {"g1::pass", "g2::pass"}


def test_compute_learnability_coverage_uses_union():
    contributors, uncovered = _compute_learnability_coverage(
        {"a", "b", "c"}, ("a",), ("b",)
    )
    assert contributors == {"a", "b"}
    assert uncovered == {"c"}


def test_build_current_evidence_map_dedupes_signatures(tmp_path):
    pool = HypothesisPool(
        pool_path=tmp_path / "hypotheses.json",
        events_path=tmp_path / "hypothesis_events.jsonl",
        min_support_cycles=2,
        min_unique_ticket_keys=2,
    )
    hyp1 = HypothesisCandidate(
        text="若能确认关键点齐全则通过，否则不通过。",
        falsifier="若关键点缺失则不通过。",
        evidence=("g1::pass",),
    )
    hyp2 = HypothesisCandidate(
        text="若能确认关键点齐全则通过，否则不通过。",
        falsifier="若关键点缺失则不通过。",
        evidence=("g2::pass",),
    )
    evidence_map = pool.build_current_evidence_map([hyp1, hyp2])
    assert len(evidence_map) == 1
    evidence = next(iter(evidence_map.values()))
    assert set(evidence) == {"g1::pass", "g2::pass"}


def test_mark_rejected_updates_status(tmp_path):
    pool = HypothesisPool(
        pool_path=tmp_path / "hypotheses.json",
        events_path=tmp_path / "hypothesis_events.jsonl",
        min_support_cycles=2,
        min_unique_ticket_keys=2,
    )
    hyp = HypothesisCandidate(
        text="若能确认关键点齐全则通过，否则不通过。",
        falsifier="若关键点缺失则不通过。",
        evidence=("g1::pass",),
    )
    pool.record_proposals([hyp], reflection_cycle=0, epoch=1, allow_promote=False)
    signature = next(iter(pool.load().keys()))
    rejected = pool.mark_rejected(
        [signature], reflection_cycle=1, epoch=1, reason="invalid"
    )
    assert rejected and rejected[0].status == "rejected"
    reloaded = pool.load()[signature]
    assert reloaded.status == "rejected"
