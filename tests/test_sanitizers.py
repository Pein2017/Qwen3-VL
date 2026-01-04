from data_conversion.utils.sanitizers import (
    fold_free_text_into_remark,
    sanitize_free_text_value,
    sanitize_station_distance_value,
)


def test_sanitize_station_distance_value_digits() -> None:
    assert sanitize_station_distance_value("373、") == "373"
    assert sanitize_station_distance_value("  67 ") == "67"
    assert sanitize_station_distance_value("距离:128m") == "128"


def test_sanitize_station_distance_value_unreadable() -> None:
    assert sanitize_station_distance_value("无法识别") is None
    assert sanitize_station_distance_value("看不清") is None
    assert sanitize_station_distance_value("") is None
    assert sanitize_station_distance_value(None) is None


def test_sanitize_free_text_value_preserves_symbols() -> None:
    assert sanitize_free_text_value(" A ,B|C=1 ") == "A,B|C=1"


def test_sanitize_free_text_value_drops_learning_note() -> None:
    text = "备注1,这里已经帮助修改,请注意参考学习,备注2"
    assert sanitize_free_text_value(text) == "备注1,备注2"


def test_sanitize_free_text_value_drops_noise_tokens() -> None:
    text = "备注1,请参考学习,备注2,建议看下操作手册中螺丝、插头的标注规范"
    assert sanitize_free_text_value(text) == "备注1,备注2"


def test_fold_free_text_into_remark() -> None:
    desc = "类别=BBU设备,品牌=华为,可见性=完整,挡风板需求=免装,无法判断品牌"
    assert (
        fold_free_text_into_remark(desc)
        == "类别=BBU设备,品牌=华为,可见性=完整,挡风板需求=免装,备注=无法判断品牌"
    )


def test_fold_free_text_into_remark_keeps_ocr_commas() -> None:
    desc = "类别=标签,文本=中国联通,坪地横坑仔"
    assert fold_free_text_into_remark(desc) == "类别=标签,文本=中国联通,坪地横坑仔"
