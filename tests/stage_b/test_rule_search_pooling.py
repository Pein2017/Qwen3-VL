from __future__ import annotations

from src.stage_b.runner import _resolve_train_pool_size, _sample_train_pool_tickets
from src.stage_b.types import GroupTicket, StageASummaries


def _ticket(i: int, *, label: str) -> GroupTicket:
    return GroupTicket(
        group_id=f"QC-POOL-{i:04d}",
        mission="挡风板安装检查",
        label=label,  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "dummy"}),
        uid=f"QC-POOL-{i:04d}::{label}",
    )


def test_resolve_train_pool_size_caps_to_total() -> None:
    assert (
        _resolve_train_pool_size(
            total=10, train_pool_size=512, train_pool_fraction=None
        )
        == 10
    )
    assert (
        _resolve_train_pool_size(total=10, train_pool_size=5, train_pool_fraction=None)
        == 5
    )
    assert (
        _resolve_train_pool_size(total=10, train_pool_size=5, train_pool_fraction=0.2)
        == 2
    )


def test_sample_train_pool_tickets_with_replacement_caps() -> None:
    tickets = [_ticket(i, label="pass") for i in range(5)]
    sampled = _sample_train_pool_tickets(
        tickets,
        pool_size=10,
        with_replacement=True,
        seed=7,
    )
    assert len(sampled) == 5
