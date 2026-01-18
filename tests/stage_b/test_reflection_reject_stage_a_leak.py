from src.stage_b.reflection.engine import ReflectionEngine


def test_reject_stage_a_object_chain_without_multiplier():
    # Regression: 8B may paste Stage-A object-chain summary into guidance text.
    leaked = "BBU设备/完整,空间充足需安装/按要求配备。"
    assert ReflectionEngine._reject_experience_text(leaked) is True


def test_reject_stage_a_two_line_json_summary_leak():
    leaked = '<DOMAIN=BBU>, <TASK=SUMMARY>\n{"统计":[]}'
    assert ReflectionEngine._reject_experience_text(leaked) is True


def test_allow_general_natural_language_rule():
    ok = "若全局图确认机柜空间充足且需要挡风板，则必须在任一图片中看到挡风板已安装且方向正确；否则不通过。"
    assert ReflectionEngine._reject_experience_text(ok) is False
