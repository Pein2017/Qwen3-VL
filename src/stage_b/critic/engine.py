#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CriticEngine: LLM-based per-candidate evaluation with strict-JSON output."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Sequence, Tuple, Literal

from ..config import CriticConfig
from ..types import CriticOutput, DeterministicSignals, ParsedTrajectory, ChineseVerdict

logger = logging.getLogger(__name__)


class CriticEngine:
    """
    LLM-based critic that generates per-candidate evaluations with strict-JSON output.
    
    Replaces the rule-based SampleSummarizer with LLM-driven evaluation that produces:
    - summary: concise description of candidate behavior
    - critique: analysis of issues or strengths
    - root_cause: optional root cause analysis
    - issues: optional list of specific problems
    - candidate_ops: optional suggested operations (pooled, not applied directly)
    - uncertainty_note: optional note about evaluation uncertainty
    """

    def __init__(
        self,
        config: CriticConfig,
        model: Any,
        tokenizer: Any,
        processor: Optional[Any] = None,
    ):
        """
        Initialize CriticEngine with shared model instance.
        
        Args:
            config: CriticConfig with prompt path, generation params, length caps
            model: Shared Qwen3-VL model instance
            processor: Shared processor instance
            tokenizer: Shared tokenizer instance
        """
        self.config = config
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        # Load and validate prompt template
        if not config.prompt_path.exists():
            raise FileNotFoundError(f"Critic prompt template not found: {config.prompt_path}")
        
        with config.prompt_path.open("r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        
        # Validate template has required placeholders
        self._validate_template()

        logger.info(
            f"CriticEngine initialized: enabled={config.enabled}, "
            f"max_candidates={config.max_candidates}, "
            f"temperature={config.temperature}"
        )

    def _validate_template(self) -> None:
        """Validate that prompt template contains required placeholders."""
        required_placeholders = [
            "{mission}",
            "{stage_a_summary}",
            "{candidate_response}",
            "{signals}",
        ]
        
        missing = [p for p in required_placeholders if p not in self.prompt_template]
        if missing:
            raise ValueError(
                f"Critic prompt template missing required placeholders: {missing}"
            )
        
        # Check for strict-JSON instructions
        if "json" not in self.prompt_template.lower():
            import warnings
            warnings.warn(
                "Critic prompt template should mention strict JSON output",
                UserWarning,
                stacklevel=2,
            )

    def _prefilter_candidates(
        self,
        candidates: List[ParsedTrajectory],
        signals: List[DeterministicSignals],
    ) -> List[int]:
        """
        Pre-filter candidates to select up to max_candidates for evaluation.
        
        Prioritizes candidates with:
        - Label mismatch (label_match=False)
        - Low self-consistency
        - Contradictions across candidates (mixed label_match)

        Args:
            candidates: List of parsed trajectories
            signals: List of deterministic signals (parallel to candidates)
        
        Returns:
            List of candidate indices to evaluate (up to max_candidates)
        """
        if len(candidates) <= self.config.max_candidates:
            return list(range(len(candidates)))
        
        # Score candidates by priority
        scored: List[Tuple[int, float]] = []
        # Detect contradictions from label_match spread across candidates
        mixed = any(s.label_match is True for s in signals) and any(s.label_match is False for s in signals)
        for idx, sig in enumerate(signals):
            score = 0.0

            # Prioritize label mismatches
            if sig.label_match is False:
                score += 10.0

            # Prioritize low self-consistency
            if sig.self_consistency is not None:
                score += (1.0 - sig.self_consistency) * 5.0

            # Prioritize contradictions (mixed label_match across the group)
            if mixed:
                score += 3.0

            scored.append((idx, score))

        # Sort by score descending and take top max_candidates
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in scored[: self.config.max_candidates]]
        selected_indices.sort()  # Maintain original order

        return selected_indices

    def _build_critic_prompt(
        self,
        mission: str,
        stage_a_summary: str,
        candidate: ParsedTrajectory,
        signals: DeterministicSignals,
    ) -> str:
        """
        Build critic prompt for a single candidate.
        
        Args:
            mission: Mission identifier
            stage_a_summary: Stage-A summary text
            candidate: Parsed trajectory to evaluate
            signals: Deterministic signals for this candidate
        
        Returns:
            Formatted prompt string
        """
        # Format signals as readable text
        def _fmt_float(value: Optional[float]) -> str:
            return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"

        signals_text = (
            f"Label Match: {signals.label_match}\n"
            f"Self-Consistency: {_fmt_float(signals.self_consistency)}\n"
            f"Confidence: {_fmt_float(signals.confidence)}"
        )
        return self._safe_format_template(
            self.prompt_template,
            mission=mission,
            stage_a_summary=stage_a_summary,
            candidate_response=candidate.base.response_text,
            signals=signals_text,
        )

    def _safe_format_template(self, template: str, **kwargs: Any) -> str:
        """
        Safely format a template that may include literal JSON braces by
        escaping all braces then restoring known placeholders.
        """
        escaped = template.replace("{", "{{").replace("}", "}}")
        for key in ("mission", "stage_a_summary", "candidate_response", "signals"):
            escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
        return escaped.format(**kwargs)

    def _generate_critic_response(self, prompt: str) -> str:
        """
        Generate LLM response for critic evaluation.
        
        Args:
            prompt: Formatted critic prompt
        
        Returns:
            Raw LLM response text
        """
        # Prepare inputs
        if self.processor is None:
            raise RuntimeError("CriticEngine processor is not configured")
        inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate with configured parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()

    def _parse_critic_json(self, response: str, group_id: str) -> CriticOutput:
        """
        Parse LLM response as strict JSON according to P1.11 schema.

        Args:
            response: Raw LLM response
            group_id: Group ID for error context

        Returns:
            Parsed CriticOutput

        Raises:
            ValueError: If JSON is malformed or missing required fields
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Critic response for {group_id} is not valid JSON: {e}\n"
                f"Response: {response[:200]}"
            )

        # Validate required fields
        if "summary" not in data:
            raise ValueError(f"Critic response for {group_id} missing 'summary' field")
        if "critique" not in data:
            raise ValueError(f"Critic response for {group_id} missing 'critique' field")

        # Extract optional fields
        root_cause = data.get("root_cause")
        issues_raw = data.get("issues")
        issues = tuple(issues_raw) if isinstance(issues_raw, list) else None

        # Deprecated; accepted for backward compatibility but not exported
        candidate_ops_raw = data.get("candidate_ops")
        candidate_ops = (
            tuple(candidate_ops_raw) if isinstance(candidate_ops_raw, list) else None
        )

        uncertainty_note = data.get("uncertainty_note")

        # P1.11 LLM-only risk/uncertainty fields (Chinese values)
        needs_recheck = data.get("needs_recheck")
        uncertainty_reason = data.get("uncertainty_reason")
        evidence_sufficiency = data.get("evidence_sufficiency")
        suspected_label_noise = data.get("suspected_label_noise")

        # Basic type normalization (do not coerce aggressively)
        def _str_or_none(x):
            return str(x) if isinstance(x, str) and x.strip() else None
        def _bool_or_none(x):
            return bool(x) if isinstance(x, bool) else None
        # Narrow to expected literal sets
        verdict_value: Optional["ChineseVerdict"]
        verdict_raw = data.get("verdict")
        if isinstance(verdict_raw, str) and verdict_raw in {"通过", "不通过"}:
            verdict_value = verdict_raw  # type: ignore[assignment]
        else:
            verdict_value = None
        eq_level_raw = data.get("evidence_quality_level")
        if isinstance(eq_level_raw, str) and eq_level_raw in {"高", "中", "低"}:
            eq_level_value: Optional["Literal['高','中','低']"] = eq_level_raw  # type: ignore[assignment]
        else:
            eq_level_value = None
        label_cons_raw = data.get("label_consistency")
        if isinstance(label_cons_raw, str) and label_cons_raw in {"一致", "矛盾", "不确定"}:
            label_cons_value: Optional["Literal['一致','矛盾','不确定']"] = label_cons_raw  # type: ignore[assignment]
        else:
            label_cons_value = None
        rec_action_raw = data.get("recommended_action")
        if isinstance(rec_action_raw, str) and rec_action_raw in {"通过", "不通过", "人工复核"}:
            rec_action_value: Optional["Literal['通过','不通过','人工复核']"] = rec_action_raw  # type: ignore[assignment]
        else:
            rec_action_value = None

        return CriticOutput(
            summary=str(data["summary"]),
            critique=str(data["critique"]),
            root_cause=str(root_cause) if _str_or_none(root_cause) else None,
            issues=issues,
            candidate_ops=candidate_ops,
            uncertainty_note=_str_or_none(uncertainty_note),
            verdict=verdict_value,
            needs_recheck=_bool_or_none(needs_recheck),
            uncertainty_reason=_str_or_none(uncertainty_reason),
            evidence_quality_level=eq_level_value,
            evidence_sufficiency=_bool_or_none(evidence_sufficiency),
            label_consistency=label_cons_value,
            suspected_label_noise=_bool_or_none(suspected_label_noise),
            recommended_action=rec_action_value,
        )

    def _enforce_length_caps(self, critic: CriticOutput) -> CriticOutput:
        """
        Enforce length caps on summary and critique fields.
        
        Args:
            critic: Original CriticOutput
        
        Returns:
            CriticOutput with truncated fields if necessary
        """
        summary = critic.summary
        critique = critic.critique
        
        truncated = False
        
        if len(summary) > self.config.summary_max_chars:
            summary = summary[: self.config.summary_max_chars]
            truncated = True
        
        if len(critique) > self.config.critique_max_chars:
            critique = critique[: self.config.critique_max_chars]
            truncated = True
        
        if truncated:
            logger.debug(
                f"Truncated critic fields: summary={len(critic.summary)}->{len(summary)}, "
                f"critique={len(critic.critique)}->{len(critique)}"
            )
        
        return CriticOutput(
            summary=summary,
            critique=critique,
            root_cause=critic.root_cause,
            issues=critic.issues,
            candidate_ops=critic.candidate_ops,
            uncertainty_note=critic.uncertainty_note,
            verdict=critic.verdict,
            needs_recheck=critic.needs_recheck,
            uncertainty_reason=critic.uncertainty_reason,
            evidence_quality_level=critic.evidence_quality_level,
            evidence_sufficiency=critic.evidence_sufficiency,
            label_consistency=critic.label_consistency,
            suspected_label_noise=critic.suspected_label_noise,
            recommended_action=critic.recommended_action,
        )

    def evaluate(
        self,
        group_id: str,
        mission: str,
        candidates: Sequence[ParsedTrajectory],
        signals: Sequence[DeterministicSignals],
        stage_a_summary: str,
    ) -> List[Optional[CriticOutput]]:
        """
        Evaluate candidates and generate per-candidate critic outputs.
        
        Args:
            group_id: Group identifier for logging
            mission: Mission identifier
            candidates: List of parsed trajectories
            signals: List of deterministic signals (parallel to candidates)
            stage_a_summary: Stage-A summary text
        
        Returns:
            List of CriticOutput (parallel to candidates), None for non-evaluated candidates
        """
        if not self.config.enabled:
            return [None] * len(candidates)
        
        # Pre-filter candidates
        selected_indices = self._prefilter_candidates(candidates, signals)
        
        logger.debug(
            f"CriticEngine evaluating {len(selected_indices)}/{len(candidates)} "
            f"candidates for group {group_id}"
        )
        
        # Initialize results (None for non-selected candidates)
        results: List[Optional[CriticOutput]] = [None] * len(candidates)
        
        # Evaluate selected candidates
        for idx in selected_indices:
            try:
                # Build prompt
                prompt = self._build_critic_prompt(
                    mission=mission,
                    stage_a_summary=stage_a_summary,
                    candidate=candidates[idx],
                    signals=signals[idx],
                )
                
                # Generate response
                response = self._generate_critic_response(prompt)
                
                # Parse JSON
                critic = self._parse_critic_json(response, group_id)
                
                # Enforce length caps
                critic = self._enforce_length_caps(critic)
                
                results[idx] = critic
                
            except Exception as e:
                logger.error(
                    f"CriticEngine failed for group {group_id}, candidate {idx}: {e}",
                    exc_info=True,
                )
                # Leave as None on error
        
        return results

