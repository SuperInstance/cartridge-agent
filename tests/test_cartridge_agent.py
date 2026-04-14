#!/usr/bin/env python3
"""
Cartridge Agent — Test Suite

Tests for the core cartridge system, bridge, scene manager,
and cartridge builder.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from typing import Any, Dict

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartridge import (
    BUILTIN_CARTRIDGES,
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeState,
    CartridgeTool,
    Formality,
)
from bridge import (
    CartridgeBridge,
    CommandContext,
    CommandResult,
    Room,
    Skin,
)
from scene import Scene, SceneTransition, SceneManager
from cartridge_builder import (
    CartridgeBuilder,
    CartridgeManifest,
    TEMPLATE_NAMES,
    build_from_template,
    load_cartridge_from_file,
    package_cartridge,
    run_cartridge_tests,
)


# ======================================================================
# Test: Cartridge Core
# ======================================================================

class TestCartridgeMetadata(unittest.TestCase):
    """Tests for CartridgeMetadata."""

    def test_basic_metadata(self) -> None:
        meta = CartridgeMetadata(
            name="test-cart",
            version="1.0.0",
            description="A test cartridge",
        )
        self.assertEqual(meta.name, "test-cart")
        self.assertEqual(meta.version, "1.0.0")
        self.assertEqual(meta.capabilities, [])

    def test_checksum_deterministic(self) -> None:
        meta = CartridgeMetadata(name="test", version="1.0.0")
        c1 = meta.compute_checksum()
        c2 = meta.compute_checksum()
        self.assertEqual(c1, c2)

    def test_checksum_changes(self) -> None:
        meta1 = CartridgeMetadata(name="test", version="1.0.0")
        meta2 = CartridgeMetadata(name="test", version="2.0.0")
        self.assertNotEqual(meta1.compute_checksum(), meta2.compute_checksum())

    def test_round_trip(self) -> None:
        meta = CartridgeMetadata(
            name="rt-cart",
            version="3.2.1",
            description="Round trip test",
            capabilities=["a", "b"],
            trust_threshold=0.5,
        )
        d = meta.to_dict()
        restored = CartridgeMetadata.from_dict(d)
        self.assertEqual(restored.name, meta.name)
        self.assertEqual(restored.version, meta.version)
        self.assertEqual(restored.capabilities, meta.capabilities)
        self.assertAlmostEqual(restored.trust_threshold, 0.5)


class TestCartridgeLifecycle(unittest.TestCase):
    """Tests for Cartridge lifecycle transitions."""

    def _make_cartridge(self) -> Cartridge:
        meta = CartridgeMetadata(name="lc-test", version="1.0.0")
        cart = Cartridge(meta, [
            CartridgeTool("echo", "Echo tool"),
            CartridgeTool("add", "Add tool"),
        ])
        return cart

    def test_initial_state(self) -> None:
        cart = self._make_cartridge()
        self.assertEqual(cart.state, CartridgeState.UNLOADED)
        self.assertFalse(cart.is_active)
        self.assertFalse(cart.is_loaded)

    def test_load(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        self.assertEqual(cart.state, CartridgeState.LOADED)
        self.assertTrue(cart.is_loaded)
        self.assertFalse(cart.is_active)

    def test_activate(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        self.assertEqual(cart.state, CartridgeState.ACTIVE)
        self.assertTrue(cart.is_active)

    def test_deactivate(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        cart.deactivate()
        self.assertEqual(cart.state, CartridgeState.LOADED)

    def test_unload(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.unload()
        self.assertEqual(cart.state, CartridgeState.UNLOADED)

    def test_unload_deactivates_first(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        cart.unload()
        self.assertEqual(cart.state, CartridgeState.UNLOADED)

    def test_cannot_load_active(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        with self.assertRaises(RuntimeError):
            cart.load()

    def test_cannot_activate_unloaded(self) -> None:
        cart = self._make_cartridge()
        with self.assertRaises(RuntimeError):
            cart.activate()

    def test_hot_swap(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        new_meta = CartridgeMetadata(name="lc-test", version="2.0.0")
        swapped = cart.hot_swap(new_meta)
        self.assertTrue(swapped)
        self.assertEqual(cart.version, "2.0.0")
        self.assertEqual(cart.state, CartridgeState.LOADED)

    def test_hot_swap_preserves_active(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        new_meta = CartridgeMetadata(name="lc-test", version="2.0.0")
        swapped = cart.hot_swap(new_meta)
        self.assertTrue(swapped)
        self.assertTrue(cart.is_active)

    def test_hot_swap_same_checksum(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        same_meta = CartridgeMetadata(name="lc-test", version="1.0.0")
        swapped = cart.hot_swap(same_meta)
        self.assertFalse(swapped)

    def test_set_error(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.set_error("something broke")
        self.assertEqual(cart.state, CartridgeState.ERROR)
        self.assertEqual(cart.error, "something broke")

    def test_execute_with_handler(self) -> None:
        cart = self._make_cartridge()
        cart.add_tool(CartridgeTool(
            "double", "Double a number",
            handler=lambda x: x * 2,
        ))
        cart.load()
        cart.activate()
        result = cart.execute("double", 5)
        self.assertEqual(result, 10)
        self.assertEqual(cart._execution_count, 1)

    def test_execute_unknown_command(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        cart.activate()
        with self.assertRaises(KeyError):
            cart.execute("nonexistent")

    def test_add_remove_tool(self) -> None:
        cart = self._make_cartridge()
        cart.add_tool(CartridgeTool("new_tool", "A new tool"))
        self.assertIn("new_tool", cart.tools)
        cart.remove_tool("new_tool")
        self.assertNotIn("new_tool", cart.tools)

    def test_status_line(self) -> None:
        cart = self._make_cartridge()
        line = cart.status_line()
        self.assertIn("lc-test", line)
        self.assertIn("unloaded", line)

    def test_serialization(self) -> None:
        cart = self._make_cartridge()
        cart.load()
        d = cart.to_dict()
        self.assertEqual(d["metadata"]["name"], "lc-test")
        self.assertEqual(d["state"], "loaded")
        self.assertEqual(len(d["tools"]), 2)


class TestCartridgeRegistry(unittest.TestCase):
    """Tests for CartridgeRegistry."""

    def test_register_and_get(self) -> None:
        reg = CartridgeRegistry()
        cart = Cartridge(CartridgeMetadata(name="reg-test", version="1.0.0"))
        reg.register(cart)
        self.assertIs(reg.get("reg-test"), cart)

    def test_unregister(self) -> None:
        reg = CartridgeRegistry()
        cart = Cartridge(CartridgeMetadata(name="unreg-test"))
        reg.register(cart)
        removed = reg.unregister("unreg-test")
        self.assertIs(removed, cart)
        self.assertIsNone(reg.get("unreg-test"))

    def test_list_all(self) -> None:
        reg = CartridgeRegistry()
        reg.register(Cartridge(CartridgeMetadata(name="a")))
        reg.register(Cartridge(CartridgeMetadata(name="b")))
        self.assertEqual(len(reg.list_all()), 2)

    def test_list_by_capability(self) -> None:
        reg = CartridgeRegistry()
        reg.register(Cartridge(CartridgeMetadata(
            name="cap-test", capabilities=["xray", "yankee"],
        )))
        found = reg.list_by_capability("xray")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "cap-test")

    def test_dependency_resolution(self) -> None:
        reg = CartridgeRegistry()
        reg.register(Cartridge(CartridgeMetadata(name="a")))
        reg.register(Cartridge(CartridgeMetadata(name="b", dependencies=["a"])))
        reg.register(Cartridge(CartridgeMetadata(
            name="c", dependencies=["a", "b"],
        )))
        deps = reg.resolve_dependencies("c")
        self.assertEqual(deps, ["a", "b"])

    def test_check_missing_dependencies(self) -> None:
        reg = CartridgeRegistry()
        reg.register(Cartridge(CartridgeMetadata(
            name="orphan", dependencies=["missing"],
        )))
        missing = reg.check_dependencies("orphan")
        self.assertEqual(missing, ["missing"])

    def test_event_hooks(self) -> None:
        reg = CartridgeRegistry()
        events: list = []
        reg.on("registered", lambda c: events.append(("reg", c.name)))
        reg.register(Cartridge(CartridgeMetadata(name="hooked")))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], ("reg", "hooked"))

    def test_status_dict(self) -> None:
        reg = CartridgeRegistry()
        reg.register(Cartridge(CartridgeMetadata(name="a")))
        d = reg.to_dict()
        self.assertEqual(d["count"], 1)
        self.assertEqual(d["active_count"], 0)

    def test_builtins_load(self) -> None:
        self.assertGreater(len(BUILTIN_CARTRIDGES), 0)
        names = [c.name for c in BUILTIN_CARTRIDGES]
        self.assertIn("spreader-loop", names)
        self.assertIn("oracle-relay", names)
        self.assertIn("fleet-guardian", names)
        self.assertIn("navigation", names)


# ======================================================================
# Test: Bridge
# ======================================================================

class TestBridge(unittest.TestCase):
    """Tests for CartridgeBridge."""

    def setUp(self) -> None:
        self.reg = CartridgeRegistry()
        for cart in BUILTIN_CARTRIDGES:
            self.reg.register(cart)
        self.bridge = CartridgeBridge(registry=self.reg)

    def test_default_skins(self) -> None:
        self.assertIn("c3po", self.bridge.skins)
        self.assertIn("field-commander", self.bridge.skins)

    def test_create_room(self) -> None:
        room = self.bridge.create_room("test-room", "Test Room")
        self.assertEqual(room.room_id, "test-room")
        self.assertEqual(room.name, "Test Room")

    def test_assign_cartridge(self) -> None:
        self.bridge.create_room("nav")
        ok = self.bridge.assign_cartridge("nav", "navigation")
        self.assertTrue(ok)
        room = self.bridge.get_room("nav")
        assert room is not None
        self.assertEqual(room.cartridge_name, "navigation")

    def test_assign_cartridge_unknown(self) -> None:
        self.bridge.create_room("x")
        ok = self.bridge.assign_cartridge("x", "nonexistent")
        self.assertFalse(ok)

    def test_assign_skin(self) -> None:
        self.bridge.create_room("bridge")
        ok = self.bridge.assign_skin("bridge", "c3po")
        self.assertTrue(ok)

    def test_route_command(self) -> None:
        # Register a cartridge with a real handler
        test_cart = Cartridge(
            CartridgeMetadata(name="guardian-test", capabilities=["test"]),
            [CartridgeTool("health_check", "Check health",
                           handler=lambda: "healthy")],
        )
        test_cart.load()
        test_cart.activate()
        self.reg.register(test_cart)
        self.bridge.create_room("guardian", cartridge_name="guardian-test")
        ctx = CommandContext(room_id="guardian", command="health_check")
        result = self.bridge.route_command(ctx)
        self.assertTrue(result.success)

    def test_route_command_unknown_room(self) -> None:
        ctx = CommandContext(room_id="nonexistent", command="test")
        result = self.bridge.route_command(ctx)
        self.assertFalse(result.success)
        self.assertIn("Unknown room", result.error)

    def test_route_command_no_cartridge(self) -> None:
        self.bridge.create_room("empty")
        ctx = CommandContext(room_id="empty", command="test")
        result = self.bridge.route_command(ctx)
        self.assertFalse(result.success)
        self.assertIn("No cartridge", result.error)

    def test_route_command_interceptor(self) -> None:
        intercepted = [False]

        def interceptor(ctx: CommandContext) -> CommandResult:
            intercepted[0] = True
            return CommandResult(success=True, output="intercepted",
                                 command=ctx.command, room_id=ctx.room_id)

        self.bridge.add_interceptor(interceptor)
        self.bridge.create_room("room")
        ctx = CommandContext(room_id="room", command="test")
        result = self.bridge.route_command(ctx)
        self.assertTrue(intercepted[0])
        self.assertEqual(result.output, "intercepted")

    def test_mud_config(self) -> None:
        self.bridge.create_room("nav", cartridge_name="navigation",
                                skin_name="field-commander", model="glm-5.1")
        config = self.bridge.get_mud_config("nav")
        self.assertEqual(config["room_id"], "nav")
        self.assertIsNotNone(config["cartridge"])
        self.assertEqual(config["model"], "glm-5.1")

    def test_status(self) -> None:
        self.bridge.create_room("a")
        self.bridge.create_room("b")
        status = self.bridge.status()
        self.assertEqual(status["rooms"], 2)
        self.assertGreater(status["skins"], 0)

    def test_schedule_active(self) -> None:
        self.assertTrue(CartridgeBridge._schedule_active("always"))
        self.assertTrue(CartridgeBridge._schedule_active("nighttime")
                        or not CartridgeBridge._schedule_active("nighttime"))


# ======================================================================
# Test: Scene Manager
# ======================================================================

class TestSceneManager(unittest.TestCase):
    """Tests for SceneManager."""

    def setUp(self) -> None:
        self.reg = CartridgeRegistry()
        for cart in BUILTIN_CARTRIDGES:
            self.reg.register(cart)
        self.mgr = SceneManager(registry=self.reg)

    def test_create_scene(self) -> None:
        scene = self.mgr.create_scene(
            room_id="engine-room",
            cartridge_names=["spreader-loop"],
        )
        self.assertIsNotNone(scene)
        assert scene is not None
        self.assertEqual(scene.room_id, "engine-room")
        self.assertEqual(scene.cartridge_names, ["spreader-loop"])

    def test_create_scene_unknown_cartridge(self) -> None:
        scene = self.mgr.create_scene(
            room_id="void",
            cartridge_names=["nonexistent"],
        )
        self.assertIsNone(scene)

    def test_create_multi_cartridge_scene(self) -> None:
        scene = self.mgr.create_scene(
            room_id="ops",
            cartridge_names=["spreader-loop", "oracle-relay"],
        )
        self.assertIsNotNone(scene)
        assert scene is not None
        self.assertEqual(len(scene.cartridge_names), 2)

    def test_remove_scene(self) -> None:
        scene = self.mgr.create_scene("rm-test", ["spreader-loop"])
        assert scene is not None
        removed = self.mgr.remove_scene(scene.scene_id)
        self.assertIsNotNone(removed)
        self.assertIsNone(self.mgr.get_scene(scene.scene_id))

    def test_activate_scene(self) -> None:
        scene = self.mgr.create_scene("act-test", ["spreader-loop"])
        assert scene is not None
        ok = self.mgr.activate_scene(scene.scene_id)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.active_scene, scene)
        self.assertTrue(scene.active)

    def test_deactivate_scene(self) -> None:
        scene = self.mgr.create_scene("deact-test", ["spreader-loop"])
        assert scene is not None
        self.mgr.activate_scene(scene.scene_id)
        ok = self.mgr.deactivate_active()
        self.assertTrue(ok)
        self.assertIsNone(self.mgr.active_scene)

    def test_scene_transitions(self) -> None:
        s1 = self.mgr.create_scene("s1", ["spreader-loop"])
        s2 = self.mgr.create_scene("s2", ["oracle-relay"])
        assert s1 is not None and s2 is not None
        self.mgr.activate_scene(s1.scene_id)
        self.mgr.activate_scene(s2.scene_id, reason="testing")
        transitions = self.mgr.transitions
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[1].reason, "testing")

    def test_scene_clone(self) -> None:
        scene = self.mgr.create_scene("clone-test", ["spreader-loop"],
                                      skin_name="penn")
        assert scene is not None
        cloned = scene.clone(priority=5)
        self.assertNotEqual(cloned.scene_id, scene.scene_id)
        self.assertEqual(cloned.priority, 5)
        self.assertEqual(cloned.cartridge_names, scene.cartridge_names)

    def test_update_context(self) -> None:
        scene = self.mgr.create_scene("ctx-test", ["spreader-loop"])
        assert scene is not None
        ok = self.mgr.update_scene_context(scene.scene_id, key="value")
        self.assertTrue(ok)
        self.assertEqual(scene.context["key"], "value")

    def test_serialization(self) -> None:
        scene = self.mgr.create_scene("ser-test", ["spreader-loop"])
        assert scene is not None
        d = scene.to_dict()
        self.assertEqual(d["room_id"], "ser-test")
        restored = Scene.from_dict(d)
        self.assertEqual(restored.room_id, "ser-test")
        self.assertEqual(restored.scene_id, scene.scene_id)


# ======================================================================
# Test: Cartridge Builder
# ======================================================================

class TestCartridgeBuilder(unittest.TestCase):
    """Tests for CartridgeBuilder and template system."""

    def test_fluent_builder(self) -> None:
        cart = (CartridgeBuilder("fluent-test")
                .version("2.0.0")
                .description("Fluent builder test")
                .capability("test")
                .tool("run", "Run something")
                .build())
        self.assertEqual(cart.name, "fluent-test")
        self.assertEqual(cart.version, "2.0.0")
        self.assertEqual(cart.metadata.capabilities, ["test"])
        self.assertIn("run", cart.tools)

    def test_builder_with_handler(self) -> None:
        cart = (CartridgeBuilder("handler-test")
                .tool("triple", "Triple a value",
                      handler=lambda x: x * 3)
                .build())
        cart.activate()
        result = cart.execute("triple", 7)
        self.assertEqual(result, 21)

    def test_manifest_yaml_roundtrip(self) -> None:
        manifest = CartridgeManifest(
            name="yaml-test",
            version="1.5.0",
            description="YAML roundtrip",
            capabilities=["yaml"],
            tools=[{"name": "do_it", "description": "Do it"}],
        )
        raw = manifest.to_yaml()
        restored = CartridgeManifest.from_yaml(raw)
        self.assertEqual(restored.name, "yaml-test")
        self.assertEqual(restored.version, "1.5.0")
        self.assertEqual(restored.capabilities, ["yaml"])
        self.assertEqual(len(restored.tools), 1)

    def test_template_names(self) -> None:
        self.assertIn("iterative", TEMPLATE_NAMES)
        self.assertIn("relay", TEMPLATE_NAMES)
        self.assertIn("watchdog", TEMPLATE_NAMES)
        self.assertIn("navigation", TEMPLATE_NAMES)

    def test_build_from_template(self) -> None:
        builder = build_from_template("relay", "my-relay")
        manifest = builder.manifest()
        self.assertEqual(manifest.name, "my-relay")
        self.assertIn("relay", manifest.capabilities)
        self.assertGreater(len(manifest.tools), 0)

    def test_build_from_unknown_template(self) -> None:
        with self.assertRaises(ValueError):
            build_from_template("nonexistent", "test")

    def test_package_and_load(self) -> None:
        cart = (CartridgeBuilder("pkg-test")
                .version("1.0.0")
                .description("Packaging test")
                .tool("hello", "Say hello")
                .build())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = package_cartridge(cart, tmpdir)
            self.assertTrue(os.path.exists(path))
            loaded = load_cartridge_from_file(path)
            self.assertEqual(loaded.name, "pkg-test")
            self.assertIn("hello", loaded.tools)

    def test_test_framework(self) -> None:
        cart = (CartridgeBuilder("test-subject")
                .version("1.0.0")
                .tool("echo", "Echo", handler=lambda x=42: x)
                .build())
        result = run_cartridge_tests(cart)
        self.assertTrue(result.success)
        self.assertEqual(result.total, 5)
        self.assertEqual(result.passed, 5)

    def test_test_summary(self) -> None:
        from cartridge_builder import TestSuiteResult, TestResult
        suite = TestSuiteResult(cartridge_name="test")
        suite.add(TestResult("a", True))
        suite.add(TestResult("b", False, error="boom"))
        self.assertFalse(suite.success)
        self.assertIn("1/2", suite.summary())


# ======================================================================
# Test: CLI Integration
# ======================================================================

class TestCLI(unittest.TestCase):
    """Smoke tests for the CLI argument parsing."""

    def test_parse_list(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["list"])
        self.assertEqual(args.command, "list")

    def test_parse_load(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["load", "navigation"])
        self.assertEqual(args.command, "load")
        self.assertEqual(args.name, "navigation")

    def test_parse_build_template(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["build", "--template", "relay", "--name", "test-relay"])
        self.assertEqual(args.command, "build")
        self.assertEqual(args.template, "relay")

    def test_parse_scene_create(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "scene", "create", "--name", "ops",
            "--cartridges", "spreader-loop,oracle-relay",
        ])
        self.assertEqual(args.command, "scene")
        self.assertEqual(args.scene_command, "create")
        self.assertEqual(args.name, "ops")

    def test_parse_scene_list(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["scene", "list"])
        self.assertEqual(args.command, "scene")
        self.assertEqual(args.scene_command, "list")

    def test_parse_status(self) -> None:
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["status", "-v"])
        self.assertEqual(args.command, "status")
        self.assertTrue(args.verbose)


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
