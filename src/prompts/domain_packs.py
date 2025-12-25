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
    "客观罗列，desc 原文不改写；标签文本需完整保留，无法识别写“标签/无法识别”。\n"
    "关注点：BBU设备、挡风板、螺丝/光纤插头、光纤、电线。\n"
    "要点：BBU设备显示完整/挡风板需求；挡风板方向/遮挡；螺丝插头完整/符合（不符合写原因）；"
    "光纤保护/弯曲半径；电线捆扎。仅写可见证据，不确定写“需复核”。"
)

_RRU_BLOCK = (
    "【RRU领域提示（简版）】\n"
    "客观罗列，desc 原文不改写；标签与站点距离文本需完整保留，无法识别写“标签/无法识别”。\n"
    "关注点：RRU设备、紧固件、接地线、尾纤、标签、站点距离。\n"
    "要点：只需强调“配对关系”和“站点距离”的使用——同图优先用组N配对；跨图用站点距离标识同一安装点。\n"
    "站点距离强制输出：凡非“无关图片”，必须输出站点距离（站点距离/<数字或无法识别>×1）。\n"
    "分组前缀（组N:）保持原样；不输出BBU/机柜/挡风板/品牌/BBU端/ODF端等内容。"
)

_DOMAIN_PACKS = {
    "bbu": DomainKnowledgePack(domain="bbu", block=_BBU_BLOCK),
    "rru": DomainKnowledgePack(domain="rru", block=_RRU_BLOCK),
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
