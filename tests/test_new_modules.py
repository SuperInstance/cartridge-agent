#!/usr/bin/env python3
"""
Cartridge Agent — Tests for Slot, Loader, and HotSwapManager

Additional test suite covering the new modules.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartridge import (
    BUILTIN_CARTRIDGES,
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeState,
    CartridgeTool,
)
from slot import Slot, SlotManager, SlotState
from loader import CartridgeLoader, ValidationResult, LoadResult
from swap import HotSwapManager, SwapPhase, SwapRecord


# ======================================================================
# Test: Slot
# ======================================================================

class TestSlot(unittest.TestCase):
    """Tests for Slot lifecycle."""

    def _make_cartridge(self, name: str = "test", trust: float = 0.0) -> Cartridge:
        meta = CartridgeMetadata(name=name, trust_threshold=trust)
        return Cartridge(meta)

    def test_initial_state(self) -> None:
        slot = Slot("test-slot")
        self.assertEqual(slot.state, SlotState.EMPTY)
        self.assertTrue(slot.is_empty)
        self.assertIsNone(slot.cartridge)

    def test_insert(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        ok = slot.insert(cart)
        self.assertTrue(ok)
        self.assertEqual(slot.state, SlotState.OCCUPIED)
        self.assertTrue(slot.is_occupied)
        self.assertIs(slot.cartridge, cart)

    def test_insert_not_empty(self) -> None:
        slot = Slot("s1")
        slot.insert(self._make_cartridge())
        ok = slot.insert(self._make_cartridge("other"))
        self.assertFalse(ok)

    def test_eject(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        slot.insert(cart)
        ejected = slot.eject()
        self.assertIs(ejected, cart)
        self.assertTrue(slot.is_empty)
        self.assertIsNone(slot.cartridge)

    def test_eject_empty(self) -> None:
        slot = Slot("s1")
        self.assertIsNone(slot.eject())

    def test_eject_deactivates_cartridge(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        slot.insert(cart)
        ejected = slot.eject()
        self.assertEqual(ejected.state, CartridgeState.UNLOADED)

    def test_trust_threshold_reject(self) -> None:
        slot = Slot("s1", max_trust=0.5)
        cart = self._make_cartridge(trust=0.8)
        ok = slot.insert(cart)
        self.assertFalse(ok)

    def test_trust_threshold_accept(self) -> None:
        slot = Slot("s1", max_trust=0.8)
        cart = self._make_cartridge(trust=0.5)
        ok = slot.insert(cart)
        self.assertTrue(ok)

    def test_lock(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        slot.insert(cart)
        ok = slot.lock("processing")
        self.assertTrue(ok)
        self.assertTrue(slot.is_locked)
        self.assertEqual(slot.state, SlotState.LOCKED)

    def test_lock_empty(self) -> None:
        slot = Slot("s1")
        self.assertFalse(slot.lock())

    def test_eject_locked(self) -> None:
        slot = Slot("s1")
        slot.insert(self._make_cartridge())
        slot.lock("busy")
        self.assertIsNone(slot.eject())

    def test_force_eject_locked(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        slot.insert(cart)
        slot.lock("busy")
        ejected = slot.force_eject()
        self.assertIs(ejected, cart)
        self.assertTrue(slot.is_empty)

    def test_unlock(self) -> None:
        slot = Slot("s1")
        slot.insert(self._make_cartridge())
        slot.lock("busy")
        ok = slot.unlock()
        self.assertTrue(ok)
        self.assertEqual(slot.state, SlotState.OCCUPIED)

    def test_unlock_not_locked(self) -> None:
        slot = Slot("s1")
        self.assertFalse(slot.unlock())

    def test_swap(self) -> None:
        slot = Slot("s1")
        old = self._make_cartridge("old")
        old.load()
        slot.insert(old)
        new = self._make_cartridge("new")
        ejected = slot.swap(new)
        self.assertIs(ejected, old)
        self.assertEqual(slot.state, SlotState.OCCUPIED)
        self.assertEqual(slot.cartridge.metadata.name, "new")

    def test_swap_empty_slot(self) -> None:
        slot = Slot("s1")
        new = self._make_cartridge("new")
        ejected = slot.swap(new)
        self.assertIsNone(ejected)
        self.assertEqual(slot.cartridge.metadata.name, "new")

    def test_swap_locked(self) -> None:
        slot = Slot("s1")
        slot.insert(self._make_cartridge("old"))
        slot.lock("busy")
        ejected = slot.swap(self._make_cartridge("new"))
        self.assertIsNone(ejected)

    def test_events_logged(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge()
        slot.insert(cart)
        slot.lock("test")
        slot.unlock()
        slot.eject()
        self.assertEqual(len(slot.events), 4)
        types = [e.event_type for e in slot.events]
        self.assertEqual(types, ["insert", "lock", "unlock", "eject"])

    def test_status_dict(self) -> None:
        slot = Slot("s1")
        cart = self._make_cartridge("mine")
        slot.insert(cart)
        s = slot.status()
        self.assertEqual(s["name"], "s1")
        self.assertEqual(s["state"], "occupied")
        self.assertEqual(s["cartridge"], "mine")


class TestSlotManager(unittest.TestCase):
    """Tests for SlotManager."""

    def _make_cartridge(self, name: str = "test") -> Cartridge:
        return Cartridge(CartridgeMetadata(name=name))

    def test_create_slot(self) -> None:
        mgr = SlotManager()
        slot = mgr.create_slot("primary")
        self.assertEqual(slot.name, "primary")

    def test_get_slot(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        self.assertIsNotNone(mgr.get_slot("a"))
        self.assertIsNone(mgr.get_slot("b"))

    def test_remove_slot(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        removed = mgr.remove_slot("a")
        self.assertIsNotNone(removed)
        self.assertIsNone(mgr.get_slot("a"))

    def test_list_slots(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        mgr.create_slot("b")
        self.assertEqual(len(mgr.list_slots()), 2)

    def test_list_empty_and_occupied(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        mgr.create_slot("b")
        mgr.get_slot("a").insert(self._make_cartridge())
        self.assertEqual(len(mgr.list_empty()), 1)
        self.assertEqual(len(mgr.list_occupied()), 1)

    def test_bulk_insert(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        mgr.create_slot("b")
        results = mgr.bulk_insert({
            "a": self._make_cartridge("c1"),
            "b": self._make_cartridge("c2"),
        })
        self.assertTrue(results["a"])
        self.assertTrue(results["b"])

    def test_bulk_insert_missing_slot(self) -> None:
        mgr = SlotManager()
        results = mgr.bulk_insert({"missing": self._make_cartridge()})
        self.assertFalse(results["missing"])

    def test_bulk_eject(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        mgr.get_slot("a").insert(self._make_cartridge())
        results = mgr.bulk_eject(["a", "missing"])
        self.assertIsNotNone(results["a"])
        self.assertIsNone(results["missing"])

    def test_status(self) -> None:
        mgr = SlotManager()
        mgr.create_slot("a")
        mgr.create_slot("b")
        mgr.get_slot("a").insert(self._make_cartridge())
        s = mgr.status()
        self.assertEqual(s["total_slots"], 2)
        self.assertEqual(s["occupied"], 1)
        self.assertEqual(s["empty"], 1)


# ======================================================================
# Test: Loader
# ======================================================================

class TestCartridgeLoader(unittest.TestCase):
    """Tests for CartridgeLoader."""

    def test_validate_good_metadata(self) -> None:
        meta = CartridgeMetadata(
            name="good-cart",
            version="1.0.0",
            capabilities=["test"],
        )
        result = CartridgeLoader.validate_metadata(meta)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

    def test_validate_empty_name(self) -> None:
        meta = CartridgeMetadata(name="")
        result = CartridgeLoader.validate_metadata(meta)
        self.assertFalse(result.valid)

    def test_validate_bad_name_chars(self) -> None:
        meta = CartridgeMetadata(name="bad cart!")
        result = CartridgeLoader.validate_metadata(meta)
        self.assertFalse(result.valid)

    def test_validate_self_dependency(self) -> None:
        meta = CartridgeMetadata(name="loop", dependencies=["loop"])
        result = CartridgeLoader.validate_metadata(meta)
        self.assertFalse(result.valid)

    def test_validate_trust_out_of_range(self) -> None:
        meta = CartridgeMetadata(name="x", trust_threshold=1.5)
        result = CartridgeLoader.validate_metadata(meta)
        self.assertFalse(result.valid)

    def test_validate_warnings_no_capabilities(self) -> None:
        meta = CartridgeMetadata(name="x")
        result = CartridgeLoader.validate_metadata(meta)
        self.assertTrue(result.valid)
        self.assertGreater(len(result.warnings), 0)

    def test_validate_tools(self) -> None:
        tools = [
            CartridgeTool("a", "Tool A"),
            CartridgeTool("b", "Tool B"),
        ]
        result = CartridgeLoader.validate_tools(tools)
        self.assertTrue(result.valid)

    def test_validate_tools_duplicate(self) -> None:
        tools = [
            CartridgeTool("dup", "First"),
            CartridgeTool("dup", "Second"),
        ]
        result = CartridgeLoader.validate_tools(tools)
        self.assertFalse(result.valid)

    def test_validate_tools_empty_name(self) -> None:
        tools = [CartridgeTool("", "No name")]
        result = CartridgeLoader.validate_tools(tools)
        self.assertFalse(result.valid)

    def test_load_basic(self) -> None:
        loader = CartridgeLoader()
        result = loader.load(CartridgeMetadata(
            name="load-test",
            version="1.0.0",
            capabilities=["test"],
        ))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.cartridge)
        self.assertEqual(result.cartridge.name, "load-test")

    def test_load_and_activate(self) -> None:
        loader = CartridgeLoader()
        result = loader.load(
            CartridgeMetadata(name="active-test"),
            activate=True,
        )
        self.assertTrue(result.success)
        self.assertTrue(result.cartridge.is_active)

    def test_load_invalid_rejected(self) -> None:
        loader = CartridgeLoader()
        result = loader.load(CartridgeMetadata(name=""))
        self.assertFalse(result.success)

    def test_load_from_dict(self) -> None:
        loader = CartridgeLoader()
        result = loader.load_from_dict({
            "name": "dict-cart",
            "version": "2.0.0",
            "capabilities": ["dict"],
        })
        self.assertTrue(result.success)
        self.assertEqual(result.cartridge.version, "2.0.0")

    def test_load_from_dict_with_tools(self) -> None:
        loader = CartridgeLoader()
        result = loader.load_from_dict(
            {"name": "tools-cart", "version": "1.0.0"},
            tools=[{"name": "run", "description": "Run it"}],
        )
        self.assertTrue(result.success)
        self.assertIn("run", result.cartridge.tools)

    def test_load_from_dict_bad_data(self) -> None:
        loader = CartridgeLoader()
        result = loader.load_from_dict({})
        self.assertFalse(result.success)

    def test_load_from_file(self) -> None:
        loader = CartridgeLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({
                "name": "file-cart",
                "version": "1.0.0",
                "tools": [{"name": "do", "description": "Do it"}],
            }, f)
            f.flush()
            result = loader.load_from_file(f.name)
        os.unlink(f.name)
        self.assertTrue(result.success)
        self.assertEqual(result.cartridge.name, "file-cart")

    def test_load_from_missing_file(self) -> None:
        loader = CartridgeLoader()
        result = loader.load_from_file("/nonexistent/file.json")
        self.assertFalse(result.success)

    def test_load_from_bad_json(self) -> None:
        loader = CartridgeLoader()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{bad json")
            f.flush()
            result = loader.load_from_file(f.name)
        os.unlink(f.name)
        self.assertFalse(result.success)

    def test_load_batch(self) -> None:
        loader = CartridgeLoader()
        results = loader.load_batch([
            (CartridgeMetadata(name="batch-a"), []),
            (CartridgeMetadata(name="batch-b"), []),
        ])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))

    def test_load_batch_rollback_on_failure(self) -> None:
        loader = CartridgeLoader()
        results = loader.load_batch([
            (CartridgeMetadata(name="good"), []),
            (CartridgeMetadata(name=""), []),  # Invalid
        ])
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        # First cartridge should be rolled back
        self.assertIsNone(loader.registry.get("good"))

    def test_load_order(self) -> None:
        loader = CartridgeLoader()
        loader.load(CartridgeMetadata(name="a"))
        loader.load(CartridgeMetadata(name="b"))
        self.assertEqual(loader.load_order, ["a", "b"])

    def test_load_with_dependencies_builtin(self) -> None:
        loader = CartridgeLoader()
        results = loader.load_with_dependencies(
            CartridgeMetadata(
                name="dep-test",
                dependencies=["navigation"],
            ),
        )
        # Should have loaded navigation + dep-test
        self.assertTrue(results[-1].success)
        self.assertEqual(results[-1].cartridge.name, "dep-test")

    def test_load_with_missing_dependency(self) -> None:
        loader = CartridgeLoader()
        results = loader.load_with_dependencies(
            CartridgeMetadata(
                name="orphan-test",
                dependencies=["nonexistent-dep"],
            ),
        )
        self.assertFalse(results[-1].success)

    def test_strict_mode_rejects_warnings(self) -> None:
        loader = CartridgeLoader(strict=True)
        # No capabilities triggers a warning
        result = loader.load(CartridgeMetadata(name="strict-test"))
        self.assertFalse(result.success)

    def test_status(self) -> None:
        loader = CartridgeLoader()
        loader.load(CartridgeMetadata(name="s1"))
        s = loader.status()
        self.assertEqual(s["loaded"], 1)
        self.assertIn("s1", s["load_order"])


# ======================================================================
# Test: HotSwapManager
# ======================================================================

class TestHotSwapManager(unittest.TestCase):
    """Tests for HotSwapManager."""

    def _make_registry_with_cart(self) -> CartridgeRegistry:
        reg = CartridgeRegistry()
        cart = Cartridge(
            CartridgeMetadata(name="swap-target", version="1.0.0"),
            [CartridgeTool("ping", "Ping", handler=lambda: "pong")],
        )
        reg.register(cart)
        reg.load("swap-target")
        reg.activate("swap-target")
        return reg

    def test_simple_swap(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.swap(
            "swap-target",
            CartridgeMetadata(name="swap-target", version="2.0.0"),
        )
        self.assertEqual(record.phase, SwapPhase.COMPLETED)
        self.assertEqual(record.old_version, "1.0.0")
        self.assertEqual(record.new_version, "2.0.0")

    def test_swap_nonexistent(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.swap(
            "no-such-cart",
            CartridgeMetadata(name="no-such-cart", version="2.0.0"),
        )
        self.assertEqual(record.phase, SwapPhase.FAILED)

    def test_swap_same_checksum(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        # Same version → same checksum → rejected
        record = mgr.swap(
            "swap-target",
            CartridgeMetadata(name="swap-target", version="1.0.0"),
        )
        self.assertEqual(record.phase, SwapPhase.FAILED)

    def test_prepare_commit(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.prepare(
            "swap-target",
            CartridgeMetadata(name="swap-target", version="3.0.0"),
        )
        self.assertEqual(record.phase, SwapPhase.PREPARING)

        committed = mgr.commit(record.swap_id)
        self.assertEqual(committed.phase, SwapPhase.COMPLETED)

    def test_prepare_rollback(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.prepare(
            "swap-target",
            CartridgeMetadata(name="swap-target", version="4.0.0"),
        )
        self.assertEqual(record.phase, SwapPhase.PREPARING)

        rolled = mgr.rollback(record.swap_id)
        self.assertEqual(rolled.phase, SwapPhase.ROLLED_BACK)
        self.assertTrue(rolled.rolled_back)

    def test_commit_unknown_swap(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.commit("nonexistent")
        self.assertEqual(record.phase, SwapPhase.FAILED)

    def test_rollback_unknown_swap(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.rollback("nonexistent")
        self.assertEqual(record.phase, SwapPhase.FAILED)

    def test_swap_callback(self) -> None:
        reg = self._make_registry_with_cart()
        callbacks: list = []
        mgr = HotSwapManager(
            reg,
            on_swap=lambda r: callbacks.append(("swap", r.target_name)),
        )
        mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="2.0.0",
        ))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0], ("swap", "swap-target"))

    def test_rollback_callback(self) -> None:
        reg = self._make_registry_with_cart()
        callbacks: list = []
        mgr = HotSwapManager(
            reg,
            on_rollback=lambda r: callbacks.append(("rollback", r.target_name)),
        )
        record = mgr.prepare(
            "swap-target",
            CartridgeMetadata(name="swap-target", version="5.0.0"),
        )
        mgr.rollback(record.swap_id)
        self.assertEqual(len(callbacks), 1)

    def test_pending(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        mgr.prepare("swap-target", CartridgeMetadata(
            name="swap-target", version="6.0.0",
        ))
        self.assertEqual(len(mgr.get_pending()), 1)

    def test_history(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="2.0.0",
        ))
        mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="3.0.0",
        ))
        history = mgr.get_history()
        self.assertEqual(len(history), 2)

    def test_stats(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="2.0.0",
        ))
        stats = mgr.get_stats()
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertGreater(stats["avg_duration_ms"], 0)

    def test_duration_ms(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="2.0.0",
        ))
        self.assertIsNotNone(record.duration_ms)
        self.assertGreater(record.duration_ms, 0)

    def test_swap_record_serialization(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.swap("swap-target", CartridgeMetadata(
            name="swap-target", version="2.0.0",
        ))
        d = record.to_dict()
        self.assertEqual(d["target_name"], "swap-target")
        self.assertEqual(d["phase"], "completed")
        self.assertIn("duration_ms", d)

    def test_invalid_new_metadata(self) -> None:
        reg = self._make_registry_with_cart()
        mgr = HotSwapManager(reg)
        record = mgr.swap(
            "swap-target",
            CartridgeMetadata(name=""),  # Invalid name
        )
        self.assertEqual(record.phase, SwapPhase.FAILED)


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
