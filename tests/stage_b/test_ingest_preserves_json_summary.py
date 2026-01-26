import json
from datetime import datetime, timezone
from pathlib import Path

from src.stage_b.ingest.stage_a import ingest_stage_a
from src.stage_b.sampling.prompts import build_user_prompt
from src.stage_b.types import MissionGuidance


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def test_ingest_stage_a_preserves_json_summary_strings(tmp_path: Path) -> None:
    stage_a_path = tmp_path / "stage_a.jsonl"
    summary = '{"统计":[{"类别":"BBU设备","品牌":{"华为":1}}]}'
    _write_jsonl(
        stage_a_path,
        [
            {
                "mission": "挡风板安装检查",
                "group_id": "QC-JSON-INGEST-001",
                "label": "pass",
                "per_image": {"image_1": summary},
            }
        ],
    )

    tickets = list(ingest_stage_a([stage_a_path]))
    assert len(tickets) == 1

    guidance = MissionGuidance(
        mission="挡风板安装检查",
        experiences={"G0": "至少需要检测到BBU设备并判断挡风板是否按要求"},
        step=1,
        updated_at=datetime.now(timezone.utc),
    )
    prompt = build_user_prompt(tickets[0], guidance)

    # JSON summary should remain intact and be parsed for object-count estimation.
    assert '"统计"' in prompt
    assert "Image1(obj=1)" in prompt
