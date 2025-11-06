"""Unit tests for experience parsing logic."""

from src.stage_b.config import ReflectionConfig
from src.stage_b.reflection import ReflectionEngine


def test_parse_experiences_from_text(tmp_path):
    """Test parsing numbered experiences from reflection response text."""

    # Create a minimal ReflectionEngine instance (we only need the parsing method)
    # We'll use a mock model/tokenizer since we only test parsing
    class MockModel:
        pass

    class MockTokenizer:
        pass

    # Create prompt template file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Reflection prompt template", encoding="utf-8")

    config = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=2,
        apply_if_delta=0.05,
        allow_uncertain=True,
    )

    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore[arg-type]
        tokenizer=MockTokenizer(),  # type: ignore[arg-type]
        config=config,
        guidance_repo=None,  # type: ignore[arg-type]
    )

    # Test case 1: Standard numbered experiences
    text1 = "[G0]. 若挡风板缺失则判定不通过\n[G1]. 摘要置信度低时请返回不通过并说明原因"
    result1 = engine._parse_experiences_from_text(text1)
    assert result1 == {
        "G0": "若挡风板缺失则判定不通过",
        "G1": "摘要置信度低时请返回不通过并说明原因",
    }

    # Test case 2: Experiences with multiple lines
    text2 = "[G0]. 若挡风板缺失\n则判定不通过\n[G1]. 摘要置信度低时请返回不通过"
    result2 = engine._parse_experiences_from_text(text2)
    assert result2 == {
        "G0": "若挡风板缺失\n则判定不通过",
        "G1": "摘要置信度低时请返回不通过",
    }

    # Test case 3: Experiences with extra whitespace
    text3 = "[G0].  若挡风板缺失则判定不通过  \n[G1].  摘要置信度低时请返回不通过  "
    result3 = engine._parse_experiences_from_text(text3)
    assert result3 == {
        "G0": "若挡风板缺失则判定不通过",
        "G1": "摘要置信度低时请返回不通过",
    }

    # Test case 4: Empty text
    text4 = ""
    result4 = engine._parse_experiences_from_text(text4)
    assert result4 == {}

    # Test case 5: Text without experience patterns
    text5 = "This is just some text without experience patterns."
    result5 = engine._parse_experiences_from_text(text5)
    assert result5 == {}

    # Test case 6: Mixed content with experiences embedded
    # Note: Parser includes all lines between markers (including "More text here." and "End.")
    text6 = "Here is some text.\n[G0]. 若挡风板缺失则判定不通过\nMore text here.\n[G1]. 摘要置信度低时请返回不通过\nEnd."
    result6 = engine._parse_experiences_from_text(text6)
    assert result6 == {
        "G0": "若挡风板缺失则判定不通过\nMore text here.",
        "G1": "摘要置信度低时请返回不通过\nEnd.",
    }

    # Test case 7: Single experience
    text7 = "[G0]. 若挡风板缺失则判定不通过"
    result7 = engine._parse_experiences_from_text(text7)
    assert result7 == {"G0": "若挡风板缺失则判定不通过"}
