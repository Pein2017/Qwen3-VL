import json
from src.stage_b.io.guidance import GuidanceRepository, MissionGuidanceError


def test_legacy_schema_rejected(tmp_path):
    path = tmp_path / "guidance.json"
    payload = {
        "M1": {
            "version": 1,
            "updated_at": "2024-01-01T00:00:00",
            "guidance": {"G0": "legacy"},
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    repo = GuidanceRepository(path, retention=1)
    try:
        repo.load()
    except MissionGuidanceError as e:
        msg = str(e)
        assert "step" in msg or "experiences" in msg
    else:
        raise AssertionError("Legacy schema should be rejected")

