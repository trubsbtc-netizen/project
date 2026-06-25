"""Shared environment-variable and value-parsing helpers.

Consolidates the duplicated ``_env_bool``, ``_env_float``, ``_env_int``,
``_safe_float``, ``_utc_timestamp``, and ``finite_float`` implementations
that were independently maintained across multiple modules.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(str(value).replace(",", ""))
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def utc_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return None
    return None
