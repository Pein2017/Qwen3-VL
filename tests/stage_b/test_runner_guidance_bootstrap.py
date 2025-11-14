from pathlib import Path
import json
from datetime import datetime, timezone
import pytest
from src.stage_b.runner import _setup_mission_guidance


def test_guidance_bootstrap_requires_mission_in_startup(tmp_path: Path):
    """Test that _setup_mission_guidance raises RuntimeError when mission is missing from startup file."""
    startup_path = tmp_path / "startup.json"
    # Create a startup file without the required mission
    startup_path.write_text(json.dumps({
        "其他任务": {
            "focus": "其他任务要点",
            "step": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "experiences": {"G0": "初始经验"}
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    mission_dir = tmp_path / "run" / "BBU"

    # Should raise RuntimeError because "BBU安装方式检查" is not in startup file
    with pytest.raises(RuntimeError, match="Mission BBU安装方式检查 not found in global guidance file"):
        _setup_mission_guidance(
            startup_path=startup_path,
            mission_dir=mission_dir,
            mission="BBU安装方式检查",
            retention=3
        )


def test_guidance_bootstrap_creates_valid_section(tmp_path: Path):
    """Test that _setup_mission_guidance correctly copies mission section from startup file."""
    startup_path = tmp_path / "startup.json"
    # Create a startup file with the required mission
    startup_path.write_text(json.dumps({
        "BBU安装方式检查": {
            "focus": "BBU安装方式检查要点",
            "step": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "experiences": {"G0": "初始经验"}
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    mission_dir = tmp_path / "run" / "BBU"
    repo = _setup_mission_guidance(
        startup_path=startup_path,
        mission_dir=mission_dir,
        mission="BBU安装方式检查",
        retention=3
    )

    guidance_map = repo.load()
    assert "BBU安装方式检查" in guidance_map
    mg = guidance_map["BBU安装方式检查"]
    assert mg.step >= 1
    assert mg.experiences
    assert "G0" in mg.experiences

