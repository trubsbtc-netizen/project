from __future__ import annotations

from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None
    import json


def loads(data: str | bytes | bytearray) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    return json.loads(data)


def dumps(value: Any) -> str:
    if orjson is not None:
        return orjson.dumps(value).decode("utf-8")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

