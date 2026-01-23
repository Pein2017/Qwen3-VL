"""Schema-tree specs for summary GRPO rewards.

This module defines a lightweight, domain-specific "attribute tree" contract:

- When a category (类别) is emitted, certain attribute keys MUST also be emitted
  (schema completeness).
- Attribute keys outside the allowed set are penalized (schema drift control).

Why this exists:
Stage-B consumes Stage-A summaries as a *structured contract*. In practice, the
model may identify a category but:
- omit required keys (incomplete answers -> higher FN in Stage-B), or
- emit long-tail / unstable keys (e.g. 规格/颜色/导电材料) that create ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CategorySchema:
    """Schema requirements for a single `统计[*].类别` entry."""

    # Keys that must exist when the category exists.
    required_all: frozenset[str]
    # "OR groups": for each group, at least one key must exist.
    required_any: tuple[frozenset[str], ...]
    # Allowed keys for this category (anything else is "extra").
    allowed: frozenset[str]

    def __post_init__(self) -> None:
        # Basic sanity: required keys must be allowed keys.
        if not self.required_all.issubset(self.allowed):
            missing = sorted(self.required_all - self.allowed)
            raise ValueError(
                "CategorySchema.required_all must be subset of allowed; "
                f"missing={missing}"
            )
        for group in self.required_any:
            if not group:
                raise ValueError("CategorySchema.required_any must not contain empty")
            if not group.issubset(self.allowed):
                missing = sorted(group - self.allowed)
                raise ValueError(
                    "CategorySchema.required_any group must be subset of allowed; "
                    f"missing={missing}"
                )


@dataclass(frozen=True, slots=True)
class DomainSchema:
    """Schema requirements for a summary domain (BBU/RRU)."""

    # Allowed top-level keys in the summary JSON.
    allowed_top_level_keys: frozenset[str]
    # Per-category schema map.
    categories: dict[str, CategorySchema]

    def __post_init__(self) -> None:
        if "统计" not in self.allowed_top_level_keys:
            raise ValueError("DomainSchema.allowed_top_level_keys must include '统计'")


def _bbu_schema() -> DomainSchema:
    # Derived from the *full* training corpus:
    # `data_new_schema_center/bbu_full_1024/all_samples.jsonl` (summary field).
    #
    # IMPORTANT:
    # - For BBU, ONLY top-level `备注` is free-text and allowed outside the schema.
    # - Per-category extra keys (including per-category `备注`) are treated as drift
    #   and should be penalized.
    categories: dict[str, CategorySchema] = {
        "挡风板": CategorySchema(
            required_all=frozenset({"品牌", "可见性", "安装方向"}),
            required_any=(),
            allowed=frozenset({"品牌", "可见性", "安装方向"}),
        ),
        "BBU设备": CategorySchema(
            required_all=frozenset({"品牌", "可见性", "挡风板需求"}),
            required_any=(),
            allowed=frozenset({"品牌", "可见性", "挡风板需求"}),
        ),
        "BBU安装螺丝": CategorySchema(
            required_all=frozenset({"可见性", "符合性"}),
            required_any=(),
            allowed=frozenset({"可见性", "符合性"}),
        ),
        "BBU端光纤插头": CategorySchema(
            required_all=frozenset({"可见性", "符合性"}),
            required_any=(),
            allowed=frozenset({"可见性", "符合性"}),
        ),
        "ODF端光纤插头": CategorySchema(
            required_all=frozenset({"可见性", "符合性"}),
            required_any=(),
            allowed=frozenset({"可见性", "符合性"}),
        ),
        "机柜处接地螺丝": CategorySchema(
            required_all=frozenset({"可见性", "符合性"}),
            required_any=(),
            allowed=frozenset({"可见性", "符合性"}),
        ),
        "地排处接地螺丝": CategorySchema(
            required_all=frozenset({"可见性", "符合性"}),
            required_any=(),
            allowed=frozenset({"可见性", "符合性"}),
        ),
        "光纤": CategorySchema(
            # NOTE:
            # Real Stage-A rollouts frequently use `套管保护` or `保护` for the fiber
            # protection concept, while older corpora/configs sometimes used the
            # more generic `保护措施`. Treat these as equivalent protection keys
            # for schema completeness, but still keep the allowed set tight to
            # penalize long-tail drift keys (e.g. 套数/类型/有标签/...).
            required_all=frozenset({"弯曲半径"}),
            required_any=(frozenset({"保护措施", "套管保护", "保护"}),),
            allowed=frozenset({"弯曲半径", "保护措施", "套管保护", "保护"}),
        ),
        "电线": CategorySchema(
            required_all=frozenset({"捆扎"}),
            required_any=(),
            allowed=frozenset({"捆扎"}),
        ),
        # Labels: at least one of 文本/可读性 must be present.
        "标签": CategorySchema(
            required_all=frozenset(),
            required_any=(frozenset({"文本", "可读性"}),),
            allowed=frozenset({"文本", "可读性"}),
        ),
    }

    # BBU allows free-text top-level 备注.
    return DomainSchema(
        allowed_top_level_keys=frozenset({"统计", "备注"}),
        categories=categories,
    )


def _rru_schema() -> DomainSchema:
    # RRU is more compact: many categories are existence-only or have a single
    # stable attribute.
    categories: dict[str, CategorySchema] = {
        "站点距离": CategorySchema(
            required_all=frozenset({"站点距离"}),
            required_any=(),
            allowed=frozenset({"站点距离", "组"}),
        ),
        "RRU设备": CategorySchema(
            required_all=frozenset(),
            required_any=(),
            allowed=frozenset(),
        ),
        "紧固件": CategorySchema(
            required_all=frozenset({"安装状态"}),
            required_any=(),
            allowed=frozenset({"安装状态"}),
        ),
        "RRU接地端": CategorySchema(
            required_all=frozenset({"安装状态"}),
            required_any=(),
            allowed=frozenset({"安装状态"}),
        ),
        "地排接地端螺丝": CategorySchema(
            required_all=frozenset({"安装状态"}),
            required_any=(),
            allowed=frozenset({"安装状态"}),
        ),
        "尾纤": CategorySchema(
            required_all=frozenset({"套管保护", "标签"}),
            required_any=(),
            allowed=frozenset({"套管保护", "标签", "组"}),
        ),
        "接地线": CategorySchema(
            required_all=frozenset({"标签"}),
            required_any=(),
            allowed=frozenset({"标签", "组"}),
        ),
        "标签": CategorySchema(
            required_all=frozenset(),
            required_any=(frozenset({"文本", "可读性"}),),
            allowed=frozenset({"文本", "可读性", "组"}),
        ),
    }

    # RRU does NOT allow top-level 备注; it can include 分组统计.
    return DomainSchema(
        allowed_top_level_keys=frozenset({"统计", "分组统计"}),
        categories=categories,
    )


_DOMAIN_TO_SCHEMA: dict[str, DomainSchema] = {
    "BBU": _bbu_schema(),
    "RRU": _rru_schema(),
}


def get_domain_schema(domain_token: str | None) -> DomainSchema | None:
    """Return the domain schema for a summary domain token (BBU/RRU)."""

    if domain_token is None:
        return None
    return _DOMAIN_TO_SCHEMA.get(domain_token)


def get_summary_schema(domain_token: str | None) -> dict[str, CategorySchema]:
    """Return the per-category schema map for a summary domain token (BBU/RRU)."""

    schema = get_domain_schema(domain_token)
    return {} if schema is None else schema.categories


__all__ = [
    "CategorySchema",
    "DomainSchema",
    "get_domain_schema",
    "get_summary_schema",
]
