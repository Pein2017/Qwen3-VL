#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Stage-B training-free implementation.

Covers:
1. Verdict normalisation (English/Chinese aliases)
2. Guidance non-empty enforcement
3. Atomic snapshot rotation with simulated failure
4. JSON parse failures remain fatal with debug_info
5. Summary/critique export wiring
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.stage_b.config import ReflectionConfig
from src.stage_b.io import GuidanceRepository, MissionGuidanceError
from src.stage_b.reflection.engine import ReflectionEngine
from src.stage_b.signals import attach_signals
from src.stage_b.types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    ReflectionProposal,
    StageASummaries,
    TrajectoryWithSignals,
)
from src.stage_b.utils.verdict import normalize_verdict


# ============================================================================
# 1. Verdict Normalisation Regression Tests
# ============================================================================

def test_verdict_normalisation_english_aliases():
    """Test that English verdict aliases normalize correctly."""
    assert normalize_verdict("pass") == "pass"
    assert normalize_verdict("Pass") == "pass"
    assert normalize_verdict("PASS") == "pass"
    assert normalize_verdict("fail") == "fail"
    assert normalize_verdict("Fail") == "fail"
    assert normalize_verdict("FAIL") == "fail"


def test_verdict_normalisation_chinese_aliases():
    """Test that Chinese verdict aliases normalize correctly."""
    assert normalize_verdict("通过") == "pass"
    assert normalize_verdict("不通过") == "fail"


def test_verdict_normalisation_review_aliases():
    """Third-state / uncertain verdicts are forbidden and should not be normalized."""
    assert normalize_verdict("需复核") is None
    assert normalize_verdict("需要复核") is None
    assert normalize_verdict("无法判断") is None
    assert normalize_verdict("通过需要复核") is None
    assert normalize_verdict("通过需复核") is None


def test_verdict_normalisation_mixed_language_trajectories(tmp_path):
    """Test label_match computation with mixed English/Chinese verdicts."""
    from src.stage_b.config import SignalsConfig
    
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="pass",  # English label
        summaries=StageASummaries(per_image={"img1": "test"}),
    )
    
    # Create candidates with mixed language verdicts
    candidates = [
        TrajectoryWithSignals(
            candidate_index=0,
            verdict="通过",  # Chinese "pass"
            reason="检查通过",
            signals=DeterministicSignals(
                label_match=None,
                self_consistency=None,
                confidence=0.9,
            ),
        ),
        TrajectoryWithSignals(
            candidate_index=1,
            verdict="fail",  # English "fail"
            reason="Missing component",
            signals=DeterministicSignals(
                label_match=None,
                self_consistency=None,
                confidence=0.8,
            ),
        ),
    ]
    
    config = SignalsConfig(
        store_confidence=True,
        enable_consistency=True,
    )
    
    scored = attach_signals(ticket, candidates, config)
    
    # First candidate: "通过" (Chinese pass) should match label "pass"
    assert scored[0].signals.label_match is True
    
    # Second candidate: "fail" should NOT match label "pass"
    assert scored[1].signals.label_match is False


# ============================================================================
# 2. Guidance Non-Empty Enforcement Regression Tests
# ============================================================================

def test_guidance_non_empty_enforcement_on_load(tmp_path):
    """Test that loading guidance with empty experiences raises error."""
    guidance_path = tmp_path / "guidance.json"
    now_iso = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "挡风板安装检查": {
            "focus": "test",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {},  # Empty!
        }
    }
    
    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    
    repo = GuidanceRepository(guidance_path, retention=1)
    
    with pytest.raises(MissionGuidanceError, match="must be non-empty"):
        repo.load()


def test_guidance_non_empty_enforcement_on_remove_all(tmp_path):
    """Test that removing all experiences raises error."""
    guidance_path = tmp_path / "guidance.json"
    repo = GuidanceRepository(guidance_path, retention=1)
    guidance_path.write_text(json.dumps({
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "experiences": {"G0": "初始经验"}
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    guidance = repo.get("挡风板安装检查")
    all_keys = list(guidance.experiences.keys())
    
    # Try to remove all experiences
    operations = tuple(
        ExperienceOperation(
            op="remove",
            key=key,
            text=None,
            rationale="test removal",
            evidence=(),
        )
        for key in all_keys
    )
    
    proposal = ReflectionProposal(
        action="refine",
        summary=None,
        critique=None,
        operations=operations,
        evidence_group_ids=("QC-001",),
        uncertainty_note=None,
        text=None,
    )
    
    with pytest.raises(MissionGuidanceError, match="must be non-empty"):
        repo.apply_reflection(
            mission="挡风板安装检查",
            proposal=proposal,
            reflection_id="remove-all",
            source_group_ids=["QC-001"],
            operations=operations,
        )


# ============================================================================
# 3. Atomic Snapshot Rotation Regression Test
# ============================================================================

def test_atomic_snapshot_rotation_with_simulated_failure(tmp_path):
    """Test that write failure leaves live file intact and temp file is cleaned up."""
    guidance_path = tmp_path / "guidance.json"
    repo = GuidanceRepository(guidance_path, retention=2)
    guidance_path.write_text(json.dumps({
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "experiences": {"G0": "初始经验"}
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Capture original state
    with guidance_path.open("r", encoding="utf-8") as fh:
        original_payload = json.load(fh)
    original_step = original_payload["挡风板安装检查"]["step"]
    
    temp_path = guidance_path.with_suffix(guidance_path.suffix + ".tmp")
    
    def mock_open_fail(*args, **kwargs):
        if args and str(args[0]) == str(temp_path):
            raise OSError("Simulated write failure")
        return open(*args, **kwargs)
    
    proposal = ReflectionProposal(
        action="refine",
        summary="test",
        critique="test",
        operations=(
            ExperienceOperation(
                op="upsert",
                key="G0",
                text="This should fail",
                rationale="test",
                evidence=("QC-001",),
            ),
        ),
        evidence_group_ids=("QC-001",),
        uncertainty_note=None,
        text=None,
    )
    
    with patch("builtins.open", side_effect=mock_open_fail):
        with pytest.raises(OSError, match="Simulated write failure"):
            repo.apply_reflection(
                mission="挡风板安装检查",
                proposal=proposal,
                reflection_id="fail-test",
                source_group_ids=["QC-001"],
                operations=proposal.operations,
            )
    
    # Verify live file unchanged
    with guidance_path.open("r", encoding="utf-8") as fh:
        current = json.load(fh)
    assert current == original_payload
    assert current["挡风板安装检查"]["step"] == original_step
    
    # Verify temp file cleaned up
    assert not temp_path.exists()


# ============================================================================
# 4. JSON Parse Failures Remain Fatal with debug_info
# ============================================================================

class MockModel:
    device = "cpu"

    def generate(self, **kwargs):
        import torch

        prompt_len = int(kwargs["input_ids"].size(1))
        # Return dummy tokens: [prompt ...][generated ...]
        return torch.zeros((1, prompt_len + 8), dtype=torch.long)


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        # Minimal deterministic prompt for tests.
        return "PROMPT"

    def __call__(self, text, **kwargs):
        import torch

        n = 16
        return {
            "input_ids": torch.zeros((1, n), dtype=torch.long),
            "attention_mask": torch.ones((1, n), dtype=torch.long),
        }

    def decode(self, tokens, **kwargs):
        # Always return invalid JSON to trigger the parser failure path.
        return "This is not valid JSON at all { broken"


def test_json_parse_failure_fatal_with_debug_info(tmp_path):
    """Test that invalid JSON from reflection raises error and stores debug_info."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(
        "允许的操作: upsert|remove|merge; 每次最多提出 K 条 operations。",
        encoding="utf-8"
    )
    
    config = ReflectionConfig(
        decision_prompt_path=prompt_file,
        ops_prompt_path=prompt_file,
        batch_size=2,
        allow_uncertain=True,
        max_operations=3,
    )
    
    engine = ReflectionEngine(
        model=MockModel(),  # type: ignore
        tokenizer=MockTokenizer(),  # type: ignore
        config=config,
        guidance_repo=type("_FakeRepo", (), {"load": lambda self: {}})(),  # type: ignore
    )

    with pytest.raises(ValueError, match="No valid JSON"):
        engine._generate_json_payload(system_template="noop", user_prompt="noop")
    
    # Verify debug_info was stored
    assert engine._last_debug_info is not None
    assert "parse_error" in engine._last_debug_info
    assert "raw_response" in engine._last_debug_info
