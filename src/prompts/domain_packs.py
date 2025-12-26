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
    "输出 JSON 摘要；仅记录可见证据。\n"
    "关注点：BBU设备、挡风板、BBU安装螺丝、BBU端光纤插头、机柜处接地螺丝、光纤、电线、标签。\n"
    "标签 OCR：可读时用 文本=原文（去空格，保留 - / , | =）；不可读写 可读性=不可读。\n"
    "要点：BBU设备品牌/可见性/挡风板需求；挡风板品牌/方向；螺丝插头符合性与问题；"
    "光纤保护/弯曲半径；电线捆扎。\n"
)

_RRU_BLOCK = (
    "【RRU领域提示（简版）】\n"
    "输出 JSON 摘要；仅记录可见证据。\n"
    "关注点：RRU设备、紧固件、RRU接地端、地排接地端螺丝、尾纤、接地线、标签、站点距离。\n"
    "要点：强调配对关系与站点距离；站点距离为类别“站点距离”下的 站点距离 计数。\n"
    "分组可选：用 组=<id> 计数。"
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
