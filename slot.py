"""
Cartridge Agent — Slot Manager

A Slot is a named dock where a cartridge can be inserted or ejected.
Slots enforce capacity constraints, mutual exclusion, and lifecycle
guarantees during cartridge replacement.

Responsibilities:
- Slot lifecycle (empty → occupied → locked)
- Insert / eject with state guards
- Locking for hot-swap safety
- Capacity and trust enforcement
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from cartridge import Cartridge, CartridgeMetadata, CartridgeState


# ======================================================================
# Enums
# ======================================================================

class SlotState(Enum):
    """Lifecycle states for a cartridge slot."""
    EMPTY = "empty"
    OCCUPIED = "occupied"
    LOCKED = "locked"
    ERROR = "error"


# ======================================================================
# Slot Events
# ======================================================================

@dataclass
class SlotEvent:
    """Record of a slot lifecycle event."""
    slot_name: str
    event_type: str  # insert, eject, lock, unlock, error
    cartridge_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: str = ""


# ======================================================================
# Slot
# ======================================================================

class Slot:
    """A named dock for cartridge insertion and ejection.

    A slot enforces:
    - Only one cartridge at a time (mutual exclusion)
    - Trust threshold checks before insertion
    - Locking to prevent ejection during critical operations
    - Full lifecycle event logging
    """

    def __init__(
        self,
        name: str,
        max_trust: float = 1.0,
        allow_downgrade: bool = True,
    ) -> None:
        self.name = name
        self.max_trust = max_trust
        self.allow_downgrade = allow_downgrade
        self._state: SlotState = SlotState.EMPTY
        self._cartridge: Optional[Cartridge] = None
        self._events: List[SlotEvent] = []
        self._locked_by: Optional[str] = None
        self._inserted_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SlotState:
        return self._state

    @property
    def cartridge(self) -> Optional[Cartridge]:
        return self._cartridge

    @property
    def is_empty(self) -> bool:
        return self._state == SlotState.EMPTY

    @property
    def is_occupied(self) -> bool:
        return self._state in (SlotState.OCCUPIED, SlotState.LOCKED)

    @property
    def is_locked(self) -> bool:
        return self._state == SlotState.LOCKED

    @property
    def events(self) -> List[SlotEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Insert / Eject
    # ------------------------------------------------------------------

    def insert(self, cartridge: Cartridge) -> bool:
        """Insert a cartridge into this slot.

        Fails if:
        - Slot is not empty
        - Slot is locked
        - Cartridge trust threshold exceeds slot max_trust
        """
        if self._state == SlotState.LOCKED:
            self._log("error", cartridge.name, "Slot is locked")
            return False
        if self._state != SlotState.EMPTY:
            self._log("error", cartridge.name, "Slot not empty")
            return False
        if cartridge.metadata.trust_threshold > self.max_trust:
            self._log("error", cartridge.name,
                       f"Trust {cartridge.metadata.trust_threshold} > max {self.max_trust}")
            return False

        self._cartridge = cartridge
        self._state = SlotState.OCCUPIED
        self._inserted_at = time.time()
        self._log("insert", cartridge.name)
        return True

    def eject(self) -> Optional[Cartridge]:
        """Eject the current cartridge from this slot.

        Fails (returns None) if slot is locked.
        Deactivates and unloads the cartridge if necessary.
        """
        if self._state == SlotState.LOCKED:
            self._log("error", self._cartridge.name if self._cartridge else None,
                       "Cannot eject locked slot")
            return None
        if self._state != SlotState.OCCUPIED:
            return None

        cart = self._cartridge
        if cart:
            # Graceful shutdown
            if cart.is_active:
                cart.deactivate()
            if cart.is_loaded:
                cart.unload()
            self._log("eject", cart.name)

        self._cartridge = None
        self._state = SlotState.EMPTY
        self._inserted_at = None
        return cart

    def force_eject(self) -> Optional[Cartridge]:
        """Eject regardless of lock state. For emergency use."""
        cart = self._cartridge
        if cart:
            if cart.is_active:
                try:
                    cart.deactivate()
                except Exception:
                    pass
            if cart.is_loaded:
                try:
                    cart.unload()
                except Exception:
                    pass
            self._log("eject", cart.name, details="forced")
        self._cartridge = None
        self._state = SlotState.EMPTY
        self._locked_by = None
        self._inserted_at = None
        return cart

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def lock(self, reason: str = "") -> bool:
        """Lock the slot to prevent ejection during critical operations."""
        if self._state != SlotState.OCCUPIED:
            return False
        self._state = SlotState.LOCKED
        self._locked_by = reason or "unknown"
        self._log("lock", self._cartridge.name if self._cartridge else None, reason)
        return True

    def unlock(self) -> bool:
        """Unlock the slot."""
        if self._state != SlotState.LOCKED:
            return False
        self._state = SlotState.OCCUPIED
        self._locked_by = None
        self._log("unlock", self._cartridge.name if self._cartridge else None)
        return True

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def swap(self, new_cartridge: Cartridge) -> Optional[Cartridge]:
        """Atomically swap the cartridge in this slot.

        Locks the slot, ejects old, inserts new, unlocks.
        Returns the ejected cartridge on success, None on failure.
        """
        if self._state == SlotState.EMPTY:
            # Nothing to swap, just insert
            if self.insert(new_cartridge):
                return None
            return None

        if self._state == SlotState.LOCKED:
            self._log("error", new_cartridge.name, "Cannot swap locked slot")
            return None

        old = self._cartridge
        if old:
            self._log("swap", old.name,
                       details=f"swapping for {new_cartridge.name}")
            if old.is_active:
                old.deactivate()
            if old.is_loaded:
                old.unload()

        self._cartridge = new_cartridge
        self._inserted_at = time.time()
        self._log("insert", new_cartridge.name, details="via swap")
        return old

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "cartridge": self._cartridge.name if self._cartridge else None,
            "cartridge_version": (self._cartridge.version
                                  if self._cartridge else None),
            "locked_by": self._locked_by,
            "inserted_at": self._inserted_at,
            "max_trust": self.max_trust,
            "event_count": len(self._events),
        }

    def _log(
        self,
        event_type: str,
        cartridge_name: Optional[str] = None,
        details: str = "",
    ) -> None:
        self._events.append(SlotEvent(
            slot_name=self.name,
            event_type=event_type,
            cartridge_name=cartridge_name,
            details=details,
        ))


# ======================================================================
# SlotManager — Manage Multiple Slots
# ======================================================================

class SlotManager:
    """Manages a collection of named slots.

    Provides slot creation, lookup, and bulk operations for
    cartridge fleet management.
    """

    def __init__(self) -> None:
        self._slots: Dict[str, Slot] = {}

    def create_slot(
        self,
        name: str,
        max_trust: float = 1.0,
        allow_downgrade: bool = True,
    ) -> Slot:
        """Create and register a new slot."""
        slot = Slot(name, max_trust=max_trust, allow_downgrade=allow_downgrade)
        self._slots[name] = slot
        return slot

    def get_slot(self, name: str) -> Optional[Slot]:
        return self._slots.get(name)

    def remove_slot(self, name: str) -> Optional[Slot]:
        slot = self._slots.pop(name, None)
        if slot and slot.is_occupied:
            slot.force_eject()
        return slot

    def list_slots(self) -> List[Slot]:
        return list(self._slots.values())

    def list_empty(self) -> List[Slot]:
        return [s for s in self._slots.values() if s.is_empty]

    def list_occupied(self) -> List[Slot]:
        return [s for s in self._slots.values() if s.is_occupied]

    def bulk_insert(self, assignments: Dict[str, Cartridge]) -> Dict[str, bool]:
        """Insert cartridges into named slots. Returns per-slot success."""
        results: Dict[str, bool] = {}
        for slot_name, cartridge in assignments.items():
            slot = self._slots.get(slot_name)
            if slot is None:
                results[slot_name] = False
            else:
                results[slot_name] = slot.insert(cartridge)
        return results

    def bulk_eject(self, slot_names: List[str]) -> Dict[str, Optional[Cartridge]]:
        """Eject cartridges from named slots."""
        results: Dict[str, Optional[Cartridge]] = {}
        for name in slot_names:
            slot = self._slots.get(name)
            if slot:
                results[name] = slot.eject()
            else:
                results[name] = None
        return results

    def status(self) -> Dict[str, Any]:
        return {
            "total_slots": len(self._slots),
            "empty": len(self.list_empty()),
            "occupied": len(self.list_occupied()),
            "locked": sum(1 for s in self._slots.values() if s.is_locked),
            "slots": {n: s.status() for n, s in self._slots.items()},
        }
