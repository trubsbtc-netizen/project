"""Shared HTTP / TCP-connector helpers.

Consolidates the duplicated ``aiohttp.TCPConnector`` creation logic that
appeared in ``bot.py``, ``polymarket_bot/data/network.py``, and
``polymarket_bot/infrastructure/network.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import aiohttp


def create_tcp_connector(
    *,
    resolver: Optional[aiohttp.abc.AbstractResolver] = None,
    ssl: bool = True,
    limit: int = 40,
    limit_per_host: int = 10,
    ttl_dns_cache: int = 0,
    use_dns_cache: bool = False,
    keepalive_timeout: int = 60,
    enable_cleanup_closed: bool = True,
    force_close: bool = False,
    family: int = 0,
    **extra: Any,
) -> aiohttp.TCPConnector:
    kwargs: Dict[str, Any] = {
        "ssl": ssl,
        "limit": limit,
        "limit_per_host": limit_per_host,
        "ttl_dns_cache": ttl_dns_cache,
        "use_dns_cache": use_dns_cache,
        "enable_cleanup_closed": enable_cleanup_closed,
    }
    if resolver is not None:
        kwargs["resolver"] = resolver
    if family:
        kwargs["family"] = family
    if force_close:
        kwargs["force_close"] = force_close
    else:
        kwargs["keepalive_timeout"] = keepalive_timeout
    kwargs.update(extra)
    return aiohttp.TCPConnector(**kwargs)


NOISY_LOGGERS = ("aiohttp", "asyncio", "websockets", "urllib3")


def suppress_noisy_loggers(level: int | None = None) -> None:
    import logging

    target = level if level is not None else logging.WARNING
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(target)
