from __future__ import annotations

from src.stage_b.runner import _split_train_eval_pool
from src.stage_b.types import GroupTicket, StageASummaries


def _ticket(i: int, *, label: str) -> GroupTicket:
    return GroupTicket(
        group_id=f"QC-SPLIT-{i:04d}",
        mission="挡风板安装检查",
        label=label,  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "dummy"}),
        uid=f"QC-SPLIT-{i:04d}::{label}",
    )


def test_split_train_eval_pool_stratified_preserves_label_balance() -> None:
    tickets = []
    for i in range(50):
        tickets.append(_ticket(i, label="pass"))
    for i in range(50, 100):
        tickets.append(_ticket(i, label="fail"))

    train, holdout = _split_train_eval_pool(
        tickets,
        eval_size=20,
        seed=7,
        stratify_by_label=True,
    )
    assert len(train) + len(holdout) == len(tickets)
    assert len(holdout) == 20
    assert sum(1 for t in holdout if t.label == "pass") == 10
    assert sum(1 for t in holdout if t.label == "fail") == 10
