"""Helpers for assistant prefix formatting."""

from __future__ import annotations

from typing import Literal

_DOMAIN_TOKENS = {
    "bbu": "BBU",
    "rru": "RRU",
    "irrelevant": "IRRELEVANT",
}


def resolve_domain_token(raw: str | None) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower()
    return _DOMAIN_TOKENS.get(key)


def resolve_task_token(mode: Literal["dense", "summary"]) -> str:
    return "SUMMARY" if mode == "summary" else "DETECTION"


def build_assistant_prefix(*, fmt: str, domain: str, task: str, dataset: str) -> str:
    return fmt.format(domain=domain, task=task, dataset=dataset).strip()


__all__ = [
    "resolve_domain_token",
    "resolve_task_token",
    "build_assistant_prefix",
]
