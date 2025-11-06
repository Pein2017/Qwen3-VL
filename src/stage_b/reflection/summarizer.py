#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SampleSummarizer for generating per-candidate summaries and critiques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import ExperienceCandidate, GroupLabel, GroupTicket


@dataclass(frozen=True)
class SummaryCritique:
    """Summary and critique pair for a candidate."""

    summary: str
    critique: str


class SampleSummarizer:
    """
    Generates per-candidate summaries and critiques in Youtu-Agent style.
    
    Summary: 2-3 sentences restating what the candidate concluded.
    Critique: 1-2 sentences highlighting failure cues or follow-up guidance needs.
    """

    @staticmethod
    def summarize(
        ticket: GroupTicket,
        candidate: ExperienceCandidate,
    ) -> SummaryCritique:
        """
        Generate summary and critique for a candidate.
        
        Args:
            ticket: The group ticket being evaluated
            candidate: The candidate to summarize
            
        Returns:
            SummaryCritique with summary and critique strings
        """
        verdict = candidate.verdict or "unknown"
        reason = candidate.reason or "No reason provided"
        label_match = candidate.signals.label_match
        
        # Summary: restate what the candidate concluded
        if label_match is True:
            summary = (
                f"候选 {candidate.candidate_index} 判定为 {verdict}，"
                f"与标注标签一致。理由是：{reason[:100]}"
                f"{'...' if len(reason) > 100 else ''}"
            )
        elif label_match is False:
            summary = (
                f"候选 {candidate.candidate_index} 判定为 {verdict}，"
                f"与标注标签不一致（标注：{ticket.label}）。理由是：{reason[:100]}"
                f"{'...' if len(reason) > 100 else ''}"
            )
        else:
            summary = (
                f"候选 {candidate.candidate_index} 判定为 {verdict}。"
                f"理由是：{reason[:100]}{'...' if len(reason) > 100 else ''}"
            )
        
        # Critique: highlight failure cues or guidance needs
        if label_match is False:
            if candidate.confidence is not None and candidate.confidence < 0.5:
                critique = (
                    f"候选 {candidate.candidate_index} 判定错误且置信度较低（{candidate.confidence:.2f}），"
                    "可能存在理解偏差或关键信息遗漏。"
                )
            else:
                critique = (
                    f"候选 {candidate.candidate_index} 判定与标签不一致，"
                    "需要检查是否遗漏了关键检查点或理解有误。"
                )
        elif label_match is True:
            if candidate.confidence is not None and candidate.confidence < 0.7:
                critique = (
                    f"候选 {candidate.candidate_index} 判定正确但置信度较低（{candidate.confidence:.2f}），"
                    "建议加强相关指导以提高判断确定性。"
                )
            else:
                critique = (
                    f"候选 {candidate.candidate_index} 判定正确且置信度合理，"
                    "可作为参考范例。"
                )
        else:
            critique = (
                f"候选 {candidate.candidate_index} 判定结果需要进一步验证，"
                "标签匹配情况未知。"
            )
        
        return SummaryCritique(summary=summary, critique=critique)


__all__ = ["SampleSummarizer", "SummaryCritique"]

