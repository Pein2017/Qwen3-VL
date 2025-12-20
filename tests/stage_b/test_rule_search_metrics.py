from __future__ import annotations

from src.stage_b.rule_search import (
    bootstrap_rer_probability,
    build_gate_stats,
    build_ticket_stats,
    changed_fraction,
    compute_metrics,
    normalize_rule_signature,
    relative_error_reduction,
)


def test_normalize_rule_signature_is_stable() -> None:
    assert (
        normalize_rule_signature("如果 全局图 需要 安装，则 判定 不通过。")
        == normalize_rule_signature("若全局图需要安装则判定不通过")
    )


def test_relative_error_reduction_matches_definition() -> None:
    base = compute_metrics(
        [
            build_ticket_stats(ticket_key="a::pass", gt_label="pass", verdicts=("fail",)),
            build_ticket_stats(ticket_key="b::pass", gt_label="pass", verdicts=("pass",)),
        ]
    )
    # base acc = 0.5 => err=0.5
    new = compute_metrics(
        [
            build_ticket_stats(ticket_key="a::pass", gt_label="pass", verdicts=("pass",)),
            build_ticket_stats(ticket_key="b::pass", gt_label="pass", verdicts=("pass",)),
        ]
    )
    # new acc = 1.0 => err=0.0 => RER=(0.5-0)/0.5=1.0
    assert relative_error_reduction(base, new) == 1.0


def test_compute_metrics_rates() -> None:
    metrics = compute_metrics(
        [
            build_ticket_stats(ticket_key="a::pass", gt_label="pass", verdicts=("fail",)),
            build_ticket_stats(ticket_key="b::fail", gt_label="fail", verdicts=("pass",)),
            build_ticket_stats(ticket_key="c::pass", gt_label="pass", verdicts=("pass",)),
            build_ticket_stats(ticket_key="d::fail", gt_label="fail", verdicts=("fail",)),
        ]
    )
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.fn_rate == 0.25
    assert metrics.fp_rate == 0.25


def test_changed_fraction_counts_majority_flips() -> None:
    base = {
        "a": build_ticket_stats(ticket_key="a", gt_label="pass", verdicts=("fail",)),
        "b": build_ticket_stats(ticket_key="b", gt_label="pass", verdicts=("fail",)),
    }
    new = {
        "a": build_ticket_stats(ticket_key="a", gt_label="pass", verdicts=("pass",)),
        "b": build_ticket_stats(ticket_key="b", gt_label="pass", verdicts=("fail",)),
    }
    assert changed_fraction(base, new) == 0.5


def test_bootstrap_probability_is_one_for_clear_improvement() -> None:
    base_correct = [0] * 20
    new_correct = [1] * 20
    prob = bootstrap_rer_probability(
        base_correct,
        new_correct,
        threshold=0.1,
        iterations=100,
        seed=7,
    )
    assert prob == 1.0


def test_build_gate_stats_passes_for_clear_improvement() -> None:
    base = {}
    new = {}
    for i in range(30):
        key = f"t{i}::pass"
        base[key] = build_ticket_stats(ticket_key=key, gt_label="pass", verdicts=("fail",))
        new[key] = build_ticket_stats(ticket_key=key, gt_label="pass", verdicts=("pass",))

    stats, passed = build_gate_stats(
        base_stats=base,
        new_stats=new,
        rer_threshold=0.1,
        bootstrap_iterations=200,
        bootstrap_min_prob=0.8,
        bootstrap_seed=17,
        min_changed_fraction=0.01,
    )
    assert passed is True
    assert stats.relative_error_reduction >= 0.1
    assert stats.changed_fraction >= 0.01
    assert stats.bootstrap_prob >= 0.8
