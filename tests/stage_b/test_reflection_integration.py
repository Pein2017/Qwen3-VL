"""Integration smoke test covering reflection proposal accept/reject paths."""

import json
from datetime import datetime, timezone

from src.stage_b.config import ReflectionConfig
from src.stage_b.io import GuidanceRepository
from src.stage_b.reflection import ReflectionEngine
from src.stage_b.types import (
    DecodeConfig,
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceOperation,
    ExperienceRecord,
    GroupTicket,
    ParsedTrajectory,
    ReflectionProposal,
    StageASummaries,
    Trajectory,
    TrajectoryWithSignals,
)


class MockModel:
    """Mock model for testing reflection without actual model inference."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.device = "cpu"

    def generate(self, **kwargs):
        # Return mock token IDs that decode to response_text
        # In real implementation, this would be actual model.generate()
        return None  # Not used in our test


class MockTokenizer:
    """Mock tokenizer for testing reflection."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, **kwargs):
        # Return mock encoded inputs
        class MockEncoded:
            def __init__(self):
                self.input_ids = None
                self.attention_mask = None

            def to(self, device):
                return self

        return MockEncoded()

    def decode(self, tokens, **kwargs):
        return self.response_text


def test_reflection_refine_action_accepted(tmp_path):
    """Test that 'refine' action with valid experiences is accepted and applied."""
    guidance_path = tmp_path / "guidance.json"
    reflection_log_path = tmp_path / "reflection.jsonl"

    # Initialize guidance with step 1
    now_iso = datetime.now(timezone.utc).isoformat()
    initial_payload = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {
                "G0": "初始经验",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(initial_payload, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)

    # Create mock reflection response
    reflection_response = json.dumps(
        {
            "action": "refine",
            "text": "[G0]. 若挡风板缺失则判定不通过\n[G1]. 摘要置信度低时请返回不通过并说明原因",
            "evidence_group_ids": ["QC-001", "QC-002"],
        }
    )

    mock_model = MockModel(reflection_response)
    mock_tokenizer = MockTokenizer(reflection_response)

    # Create prompt template file BEFORE initializing ReflectionEngine
    (tmp_path / "prompt.txt").write_text("Reflection prompt template", encoding="utf-8")

    # Create engine with mocked model/tokenizer
    # We'll directly test the reflect method with a mock proposal
    engine = ReflectionEngine(
        model=mock_model,  # type: ignore[arg-type]
        tokenizer=mock_tokenizer,  # type: ignore[arg-type]
        config=ReflectionConfig(
            prompt_path=tmp_path / "prompt.txt",
            batch_size=2,
            allow_uncertain=True,
        ),
        guidance_repo=repo,
        reflection_log=reflection_log_path,
    )

    # Create minimal bundle
    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="pass",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    # Create a trajectory with signals
    trajectory = ParsedTrajectory(
        base=Trajectory(
            group_id="QC-001",
            mission="挡风板安装检查",
            candidate_index=0,
            decode=DecodeConfig(temperature=0.1, top_p=0.9, max_new_tokens=100),
            response_text="不通过\n理由: 测试\n置信度: 0.8",
            created_at=datetime.now(timezone.utc),
        ),
        verdict="不通过",  # type: ignore[arg-type]
        reason="测试",
        format_ok=True,
        confidence=0.8,
    )

    signals = DeterministicSignals(
        label_match=False,
        self_consistency=0.8,
        confidence=0.8,
    )

    candidate = TrajectoryWithSignals(parsed=trajectory, signals=signals)

    record = ExperienceRecord(
        ticket=ticket,
        candidates=(
            ExperienceCandidate(
                candidate_index=0,
                verdict="不通过",  # type: ignore[arg-type]
                reason="测试",
                confidence=0.8,
                signals=signals,
            ),
        ),
        winning_candidate=0,
        guidance_step=1,
    )

    bundle = ExperienceBundle(
        mission="挡风板安装检查",
        records=(record,),
        reflection_cycle=0,
        guidance_step=1,
    )

    # Mock the _generate_reflection to return our test proposal
    proposal = ReflectionProposal(
        action="refine",
        summary="挡板缺失仍被放行",
        critique="现有经验未覆盖挡板缺失场景",
        operations=(
            ExperienceOperation(
                op="upsert",
                key="G0",
                text="[G0]. 若挡风板缺失则判定不通过并提示复检",
                rationale=None,
                evidence=("QC-001",),
            ),
        ),
        evidence_group_ids=("QC-001",),
        uncertainty_note=None,
        text="[G0]. 若挡风板缺失则判定不通过并提示复检",
    )

    # Override _generate_reflection to return our test proposal
    original_generate = engine._generate_reflection
    engine._generate_reflection = lambda bundle: proposal  # type: ignore[assignment]

    # Execute reflection
    outcome = engine.reflect(bundle, epoch=1)
    outcome = engine.finalize_outcome(
        outcome,
        epoch=1,
        pre_uplift=0.0,
        post_uplift=0.1,
    )

    # Verify outcome
    assert outcome.applied is True
    assert outcome.eligible is True
    assert outcome.guidance_step_before == 1
    assert outcome.guidance_step_after == 2
    assert outcome.proposal.summary == "挡板缺失仍被放行"
    assert outcome.operations[0].op == "upsert"

    # Verify guidance was updated
    guidance_map = repo.load()
    updated_guidance = guidance_map["挡风板安装检查"]
    assert updated_guidance.step == 2
    assert (
        updated_guidance.experiences["G0"] == "[G0]. 若挡风板缺失则判定不通过并提示复检"
    )
    assert updated_guidance.metadata["G0"].reflection_id == outcome.reflection_id
    assert updated_guidance.metadata["G0"].sources == ("QC-001",)

    # Verify reflection log was written
    assert reflection_log_path.exists()
    log_lines = reflection_log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(log_lines) == 2
    final_log_entry = json.loads(log_lines[-1])
    assert final_log_entry["reflection"]["applied"] is True
    assert final_log_entry["reflection"]["guidance_step_before"] == 1
    assert final_log_entry["reflection"]["guidance_step_after"] == 2
    assert final_log_entry["reflection"]["proposal"]["summary"] == "挡板缺失仍被放行"


def test_reflection_noop_action_not_applied(tmp_path):
    """Test that 'noop' action is not applied."""
    guidance_path = tmp_path / "guidance.json"
    reflection_log_path = tmp_path / "reflection.jsonl"

    # Initialize guidance
    now_iso = datetime.now(timezone.utc).isoformat()
    initial_payload = {
        "挡风板安装检查": {
            "focus": "挡风板安装检查任务要点",
            "step": 1,
            "updated_at": now_iso,
            "experiences": {
                "G0": "初始经验",
            },
        }
    }

    with guidance_path.open("w", encoding="utf-8") as fh:
        json.dump(initial_payload, fh, ensure_ascii=False, indent=2)

    repo = GuidanceRepository(guidance_path, retention=1)

    # Create prompt template file BEFORE initializing ReflectionEngine
    (tmp_path / "prompt.txt").write_text("Reflection prompt template", encoding="utf-8")

    mock_model = MockModel("")
    mock_tokenizer = MockTokenizer("")

    engine = ReflectionEngine(
        model=mock_model,  # type: ignore[arg-type]
        tokenizer=mock_tokenizer,  # type: ignore[arg-type]
        config=ReflectionConfig(
            prompt_path=tmp_path / "prompt.txt",
            batch_size=2,
            allow_uncertain=True,
        ),
        guidance_repo=repo,
        reflection_log=reflection_log_path,
    )

    ticket = GroupTicket(
        group_id="QC-001",
        mission="挡风板安装检查",
        label="fail",  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "test"}),
    )

    record = ExperienceRecord(
        ticket=ticket,
        candidates=tuple(),
        winning_candidate=None,
        guidance_step=1,
    )

    bundle = ExperienceBundle(
        mission="挡风板安装检查",
        records=(record,),
        reflection_cycle=0,
        guidance_step=1,
    )

    # Mock noop proposal
    proposal = ReflectionProposal(
        action="noop",
        summary=None,
        critique=None,
        operations=(),
        evidence_group_ids=("QC-001",),
        uncertainty_note=None,
        text=None,
    )

    engine._generate_reflection = lambda bundle: proposal  # type: ignore[assignment]

    # Execute reflection
    outcome = engine.reflect(bundle, epoch=1)

    # Verify outcome
    assert outcome.applied is False
    assert outcome.guidance_step_before == 1
    assert outcome.guidance_step_after == 1  # Step should not change
    assert outcome.proposal.action == "noop"
    assert outcome.operations == tuple()

    # Verify guidance was NOT updated
    guidance_map = repo.load()
    updated_guidance = guidance_map["挡风板安装检查"]
    assert updated_guidance.step == 1  # Still at step 1
    assert updated_guidance.experiences == {"G0": "初始经验"}
