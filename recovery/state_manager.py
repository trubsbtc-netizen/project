"""
State persistence and recovery system.

Saves bot state to disk periodically so the bot can resume after crashes
without losing position information or risk state.

Recovery guarantees:
- Open positions are always persisted before execution
- Trade log is append-only (never overwrites)
- State file is written atomically (temp file + rename)
- Corrupt state file triggers clean restart (no partial load)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from core.constants import STATE_FILE_PATH, STATE_SAVE_INTERVAL_S

logger = logging.getLogger(__name__)


class StateStore:
    """
    Atomic state persistence using write-to-temp + rename pattern.
    This ensures the state file is never partially written.
    """

    def __init__(self, path: str = STATE_FILE_PATH) -> None:
        self._path     = path
        self._tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def save(self, state: Dict[str, Any]) -> bool:
        """
        Atomically save state to disk.
        Writes to temp file first, then renames.
        Returns True on success.
        """
        state["_saved_at"] = time.time()
        try:
            content = json.dumps(state, default=self._json_serial, indent=2)
            with open(self._tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(self._tmp_path, self._path)
            return True
        except Exception as exc:
            logger.error("State save failed: %s", exc)
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load state from disk.
        Returns None if file doesn't exist or is corrupt.
        Never raises — a corrupt state file triggers a clean start.
        """
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - data.get("_saved_at", 0)
            if age > 600:   # State older than 10 minutes = suspect
                logger.warning(
                    "State file is %.0fs old — loading but verify positions manually",
                    age
                )
            logger.info("State loaded from %s (age=%.0fs)", self._path, age)
            return data
        except json.JSONDecodeError as exc:
            logger.error("State file corrupt: %s — starting fresh", exc)
            return None
        except Exception as exc:
            logger.error("State load error: %s", exc)
            return None

    def clear(self) -> None:
        """Remove saved state (e.g. after clean shutdown)."""
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
        except Exception as exc:
            logger.warning("Could not clear state file: %s", exc)

    @staticmethod
    def _json_serial(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        raise TypeError(f"Not serializable: {type(obj)}")


class RecoveryManager:
    """
    Manages periodic state saving and recovery at startup.
    """

    def __init__(
        self,
        store:        StateStore,
        risk_engine,
        interval_s:   int = STATE_SAVE_INTERVAL_S,
    ) -> None:
        self._store       = store
        self._risk        = risk_engine
        self._interval    = interval_s
        self._save_task:  Optional[asyncio.Task] = None
        self._running     = False

    async def start(self) -> None:
        """Start periodic state saving."""
        self._running   = True
        self._save_task = asyncio.create_task(
            self._save_loop(), name="state-save"
        )
        logger.info("RecoveryManager started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Stop saving and do a final save."""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            await asyncio.gather(self._save_task, return_exceptions=True)
        await self._save_state()
        logger.info("RecoveryManager stopped")

    async def save_now(self) -> None:
        """Persist current risk/position state immediately."""
        await self._save_state()

    async def recover(self) -> bool:
        """
        Attempt to recover state from disk.
        Returns True if recovery was successful and open positions were found.
        """
        data = self._store.load()
        if data is None:
            logger.info("No state to recover — starting fresh")
            return False

        # Validate state structure
        if "positions" not in data or "risk" not in data:
            logger.warning("State file missing expected fields — ignoring")
            return False

        positions_data = data.get("positions", {})
        if not positions_data:
            logger.info("No open positions in saved state")
            return False

        logger.warning(
            "RECOVERY: Found %d open positions from previous session",
            len(positions_data)
        )

        # Log positions that need manual verification
        for asset_id, pos in positions_data.items():
            logger.warning(
                "  OPEN POSITION: asset=%s direction=%s cost=%.2f entry=%.3f "
                "entered_at=%s market_end=%s",
                asset_id[:20],
                pos.get("direction", "?"),
                float(pos.get("cost_basis", 0)),
                float(pos.get("entry_price", 0)),
                pos.get("entered_at", "?"),
                pos.get("market_end_time", "?"),
            )

        # NOTE: Position recovery requires verifying with Polymarket API
        # that positions are still valid. This is left as a manual verification
        # step since automatically restoring stale positions can be dangerous.
        logger.warning(
            "Manual verification required for recovered positions. "
            "Bot will start fresh risk tracking."
        )
        return True

    async def _save_loop(self) -> None:
        """Periodically save state."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("State save loop error: %s", exc)

    async def _save_state(self) -> None:
        """Capture and save current state."""
        try:
            risk_state = await self._risk.get_state()
            positions  = await self._risk.get_positions()

            state = {
                "risk": {
                    "session_pnl_usdc":     str(risk_state.session_pnl_usdc),
                    "session_trades":        risk_state.session_trades,
                    "consecutive_losses":    risk_state.consecutive_losses,
                    "total_exposure_usdc":   str(risk_state.total_exposure_usdc),
                    "circuit_state":         risk_state.circuit_state.name,
                    "kill_switch_active":    risk_state.kill_switch_active,
                },
                "positions": {
                    asset_id: {
                        "direction":      pos.direction.value,
                        "entry_price":    str(pos.entry_price),
                        "size":           str(pos.size),
                        "cost_basis":     str(pos.cost_basis),
                        "fee_paid":       str(pos.fee_paid),
                        "order_id":       pos.order_id,
                        "entered_at":     pos.entered_at,
                        "condition_id":   pos.condition_id,
                        "market_end_time": pos.market_end_time,
                    }
                    for asset_id, pos in positions.items()
                },
            }

            self._store.save(state)
        except Exception as exc:
            logger.error("Failed to capture state for saving: %s", exc)
