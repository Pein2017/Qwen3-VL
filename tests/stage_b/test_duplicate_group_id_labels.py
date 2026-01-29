import json
from pathlib import Path

from src.stage_b.ingest.stage_a import ingest_stage_a


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def test_ingest_allows_duplicate_group_id_across_labels(tmp_path: Path) -> None:
    stage_a_path = tmp_path / "stage_a.jsonl"
    group_id = "QC-TEMP-20250101-0001"
    mission = "MISSION"

    _write_jsonl(
        stage_a_path,
        [
            {
                "mission": mission,
                "group_id": group_id,
                "label": "fail",
                "per_image": {"image_1": "bad"},
            },
            {
                "mission": mission,
                "group_id": group_id,
                "label": "pass",
                "per_image": {"image_1": "good"},
            },
        ],
    )

    tickets = list(ingest_stage_a([stage_a_path]))
    assert len(tickets) == 2

    keys = {ticket.key for ticket in tickets}
    assert keys == {f"{group_id}::fail", f"{group_id}::pass"}
