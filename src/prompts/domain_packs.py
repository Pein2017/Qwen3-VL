"""Domain knowledge packs for summary inference prompts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainKnowledgePack:
    domain: str
    block: str
    is_placeholder: bool = False


_BBU_BLOCK = (
    "【BBU领域提示（简版）】\n"
    "只做客观罗列；desc 原文保持不改写。\n"
    "核心类别：BBU设备、挡风板、螺丝/光纤插头、光纤、电线、标签、需复核备注。\n"
    "常见关系：安装螺丝/光纤插头（含BBU端/ODF端/接地螺丝）、光纤保护与弯曲半径、电线捆扎、标签可识别/无法识别。"
)

_RRU_BLOCK = "【RRU领域提示】待补充"

_DOMAIN_PACKS = {
    "bbu": DomainKnowledgePack(domain="bbu", block=_BBU_BLOCK),
    "rru": DomainKnowledgePack(domain="rru", block=_RRU_BLOCK, is_placeholder=True),
}


def _normalize_domain(domain: str | None) -> str:
    if domain is None:
        return ""
    return str(domain).strip().lower()


def get_domain_pack(domain: str | None) -> DomainKnowledgePack:
    """Return the domain knowledge pack or raise for unknown values."""
    normalized = _normalize_domain(domain)
    if not normalized:
        raise ValueError("domain must be provided for runtime prompt profiles")
    pack = _DOMAIN_PACKS.get(normalized)
    if pack is None:
        available = ", ".join(sorted(_DOMAIN_PACKS.keys()))
        raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
    return pack


__all__ = ["DomainKnowledgePack", "get_domain_pack"]
