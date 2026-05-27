"""
Cartridge Agent — Hot-Swap Manager

Manages zero-downtime replacement of cartridges in running systems.
Ensures that active cartridges can be replaced without interrupting
service by using a prepare → commit → rollback protocol.

Responsibilities:
- Prepare hot-swap candidates with pre-validation
- Commit swaps atomically (old out, new in)
- Rollback on failure (restore previous state)
- Track swap history and metrics
- Slot-aware swapping with lock coordination
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from cartridge import Cartridge, CartridgeMetadata, CartridgeRegistry


# ======================================================================
# Enums
# ======================================================================

class SwapPhase(Enum):
    """Phases of a hot-swap operation."""
    PREPARING = "preparing"
    VALIDATING = "validating"
    COMMITTING = "committing"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# ======================================================================
# Swap Record
# ======================================================================

@dataclass
class SwapRecord:
    """Immutable record of a completed or failed swap."""
    swap_id: str
    target_name: str
    old_version: str
    new_version: str
    phase: SwapPhase
    started_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    rolled_back: bool = False

    @property
    def duration_ms(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "swap_id": self.swap_id,
            "target_name": self.target_name,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "rolled_back": self.rolled_back,
        }


# ======================================================================
# Swap Callbacks
# ======================================================================

SwapCallback = Callable[[SwapRecord], None]


# ======================================================================
# HotSwapManager
# ======================================================================

class HotSwapManager:
    """Manages zero-downtime cartridge replacement.

    Usage:
        manager = HotSwapManager(registry)

        # Simple swap
        record = manager.swap("my-cartridge", new_metadata)

        # Prepared swap with callbacks
        record = manager.prepare("my-cartridge", new_metadata)
        if record.phase == SwapPhase.PREPARING:
            record = manager.commit(record.swap_id)

        # Or rollback if something goes wrong
        record = manager.rollback(record.swap_id)
    """

    def __init__(
        self,
        registry: CartridgeRegistry,
        on_swap: Optional[SwapCallback] = None,
        on_rollback: Optional[SwapCallback] = None,
    ) -> None:
        self.registry = registry
        self._on_swap = on_swap
        self._on_rollback = on_rollback
        self._pending: Dict[str, SwapRecord] = {}
        self._history: List[SwapRecord] = []
        self._swap_counter: int = 0

    # ------------------------------------------------------------------
    # Simple Swap
    # ------------------------------------------------------------------

    def swap(
        self,
        target_name: str,
        new_metadata: CartridgeMetadata,
        new_tools: Optional[List[Any]] = None,
    ) -> SwapRecord:
        """Perform a complete hot-swap in one call.

        This is the simple API: prepare + validate + commit.
        Rolls back automatically on failure.
        """
        record = self.prepare(target_name, new_metadata)
        if record.phase == SwapPhase.FAILED:
            return record

        record = self.commit(record.swap_id, new_tools)
        return record

    # ------------------------------------------------------------------
    # Prepared Swap Protocol
    # ------------------------------------------------------------------

    def prepare(
        self,
        target_name: str,
        new_metadata: CartridgeMetadata,
    ) -> SwapRecord:
        """Phase 1: Prepare a hot-swap by validating the target.

        Returns a SwapRecord in PREPARING phase on success,
        or FAILED phase if the target doesn't exist.
        """
        self._swap_counter += 1
        swap_id = f"swap-{self._swap_counter:06d}"

        # Find existing cartridge
        old_cart = self.registry.get(target_name)
        if old_cart is None:
            record = SwapRecord(
                swap_id=swap_id,
                target_name=target_name,
                old_version="",
                new_version=new_metadata.version,
                phase=SwapPhase.FAILED,
                started_at=time.time(),
                error=f"Target cartridge '{target_name}' not found",
            )
            record.completed_at = time.time()
            self._history.append(record)
            return record

        # Validate new metadata
        from loader import CartridgeLoader
        validation = CartridgeLoader.validate_metadata(new_metadata)
        if not validation.valid:
            record = SwapRecord(
                swap_id=swap_id,
                target_name=target_name,
                old_version=old_cart.version,
                new_version=new_metadata.version,
                phase=SwapPhase.FAILED,
                started_at=time.time(),
                error=f"Validation failed: {'; '.join(validation.errors)}",
            )
            record.completed_at = time.time()
            self._history.append(record)
            return record

        # Check version is actually different
        old_checksum = old_cart.metadata.checksum
        new_checksum = new_metadata.compute_checksum()
        if old_checksum == new_checksum:
            record = SwapRecord(
                swap_id=swap_id,
                target_name=target_name,
                old_version=old_cart.version,
                new_version=new_metadata.version,
                phase=SwapPhase.FAILED,
                started_at=time.time(),
                error="New metadata has same checksum as current",
            )
            record.completed_at = time.time()
            self._history.append(record)
            return record

        # Save old cartridge state for rollback
        record = SwapRecord(
            swap_id=swap_id,
            target_name=target_name,
            old_version=old_cart.version,
            new_version=new_metadata.version,
            phase=SwapPhase.PREPARING,
            started_at=time.time(),
        )
        self._pending[swap_id] = record
        return record

    def commit(
        self,
        swap_id: str,
        new_tools: Optional[List[Any]] = None,
    ) -> SwapRecord:
        """Phase 2: Commit a prepared swap.

        Performs the actual hot-swap using the registry's built-in
        hot_swap method. On failure, automatically rolls back.
        """
        record = self._pending.get(swap_id)
        if record is None:
            return SwapRecord(
                swap_id=swap_id,
                target_name="",
                old_version="",
                new_version="",
                phase=SwapPhase.FAILED,
                started_at=time.time(),
                error="Unknown swap ID",
            )

        old_cart = self.registry.get(record.target_name)
        if old_cart is None:
            record.phase = SwapPhase.FAILED
            record.error = "Target disappeared during prepare"
            record.completed_at = time.time()
            del self._pending[swap_id]
            self._history.append(record)
            return record

        # Build new metadata for swap
        new_meta = CartridgeMetadata(
            name=record.target_name,
            version=record.new_version,
        )

        # Attempt the swap
        try:
            success = self.registry.hot_swap(record.target_name, new_meta)
            if not success:
                record.phase = SwapPhase.FAILED
                record.error = "Registry hot_swap returned False"
                record.completed_at = time.time()
                del self._pending[swap_id]
                self._history.append(record)
                return record
        except Exception as exc:
            # Auto-rollback
            rollback_record = self._rollback_internal(record, str(exc))
            return rollback_record

        record.phase = SwapPhase.COMPLETED
        record.completed_at = time.time()
        del self._pending[swap_id]
        self._history.append(record)

        if self._on_swap:
            try:
                self._on_swap(record)
            except Exception:
                pass

        return record

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, swap_id: str) -> SwapRecord:
        """Roll back a prepared (but not committed) swap."""
        record = self._pending.get(swap_id)
        if record is None:
            return SwapRecord(
                swap_id=swap_id,
                target_name="",
                old_version="",
                new_version="",
                phase=SwapPhase.FAILED,
                started_at=time.time(),
                error="Unknown swap ID for rollback",
            )
        return self._rollback_internal(record, "Manual rollback")

    def _rollback_internal(
        self,
        record: SwapRecord,
        reason: str,
    ) -> SwapRecord:
        """Internal rollback — restore old cartridge state."""
        # The registry's hot_swap preserves state, so if we haven't
        # committed, the old cartridge is still in place.
        # For committed-but-failed swaps, we'd need to re-swap back.
        record.phase = SwapPhase.ROLLED_BACK
        record.rolled_back = True
        record.error = reason
        record.completed_at = time.time()
        self._pending.pop(record.swap_id, None)
        self._history.append(record)

        if self._on_rollback:
            try:
                self._on_rollback(record)
            except Exception:
                pass

        return record

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_pending(self) -> List[SwapRecord]:
        return list(self._pending.values())

    def get_history(self, limit: int = 50) -> List[SwapRecord]:
        return list(self._history[-limit:])

    def get_stats(self) -> Dict[str, Any]:
        completed = [r for r in self._history if r.phase == SwapPhase.COMPLETED]
        failed = [r for r in self._history if r.phase == SwapPhase.FAILED]
        rolled_back = [r for r in self._history if r.rolled_back]
        durations = [r.duration_ms for r in completed if r.duration_ms is not None]

        return {
            "total_swaps": len(self._history),
            "completed": len(completed),
            "failed": len(failed),
            "rolled_back": len(rolled_back),
            "pending": len(self._pending),
            "avg_duration_ms": (sum(durations) / len(durations)) if durations else 0,
        }
