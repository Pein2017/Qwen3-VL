import json
from src.stage_b.critic.engine import CriticEngine
from src.stage_b.types import CriticOutput


def test_parse_critic_json_v2_fields():
    # Bypass __init__ to access parser without loading prompt/template
    engine = object.__new__(CriticEngine)
    payload = {
        "summary": "小结",
        "critique": "点评",
        "verdict": "通过",
        "needs_recheck": True,
        "evidence_sufficiency": False,
        "recommended_action": "人工复核",
    }
    out: CriticOutput = engine._parse_critic_json(json.dumps(payload), group_id="g1")
    assert out.summary == "小结"
    assert out.critique == "点评"
    assert out.verdict == "通过"
    assert out.needs_recheck is True
    assert out.evidence_sufficiency is False
    assert out.recommended_action == "人工复核"

