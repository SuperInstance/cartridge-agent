#!/usr/bin/env python3
"""
Cartridge Agent — CLI Interface

Standalone CLI for managing loadable capability modules.

Subcommands:
    serve              Start cartridge service
    list               List available cartridges
    load <name>        Load a cartridge
    unload <name>      Unload a cartridge
    build <source>     Build a cartridge from source/template
    test <name>        Test a cartridge
    scene create       Create a scene from cartridges
    scene switch       Switch active scene
    scene list         List all scenes
    onboard            Set up the agent
    status             Show agent status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cartridge import (
    BUILTIN_CARTRIDGES,
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeTool,
)
from bridge import CartridgeBridge, CommandContext, Room, Skin
from scene import SceneManager, Scene
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
# State (singleton per process)
# ======================================================================

_registry: Optional[CartridgeRegistry] = None
_bridge: Optional[CartridgeBridge] = None
_scene_manager: Optional[SceneManager] = None
_config_dir: Optional[str] = None


def _get_config_dir() -> str:
    global _config_dir
    if _config_dir is None:
        _config_dir = os.path.expanduser("~/.cartridge-agent")
    return _config_dir


def _ensure_registry() -> CartridgeRegistry:
    global _registry
    if _registry is None:
        _registry = CartridgeRegistry()
        for cart in BUILTIN_CARTRIDGES:
            _registry.register(cart)
    return _registry


def _ensure_bridge() -> CartridgeBridge:
    global _bridge
    if _bridge is None:
        _bridge = CartridgeBridge(registry=_ensure_registry())
    return _bridge


def _ensure_scene_manager() -> SceneManager:
    global _scene_manager
    if _scene_manager is None:
        _scene_manager = SceneManager(registry=_ensure_registry())
    return _scene_manager


# ======================================================================
# Helpers
# ======================================================================

def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a simple aligned table."""
    if not rows:
        print("(no data)")
        return
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep)
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


# ======================================================================
# Command Handlers
# ======================================================================

def cmd_list(args: argparse.Namespace) -> None:
    """List available cartridges."""
    reg = _ensure_registry()
    carts = reg.list_all()
    if not carts:
        print("No cartridges loaded.")
        return
    rows = []
    for c in carts:
        caps = ", ".join(c.metadata.capabilities) or "—"
        rows.append([
            c.name,
            c.version,
            c.state.value,
            str(len(c.tools)),
            caps,
        ])
    _print_table(["Name", "Version", "State", "Tools", "Capabilities"], rows)


def cmd_load(args: argparse.Namespace) -> None:
    """Load a cartridge."""
    reg = _ensure_registry()
    ok = reg.load(args.name)
    if ok:
        print(f"Loaded cartridge: {args.name}")
    else:
        print(f"Failed to load cartridge: {args.name}", file=sys.stderr)
        sys.exit(1)


def cmd_unload(args: argparse.Namespace) -> None:
    """Unload a cartridge."""
    reg = _ensure_registry()
    ok = reg.unload(args.name)
    if ok:
        print(f"Unloaded cartridge: {args.name}")
    else:
        print(f"Failed to unload cartridge: {args.name}", file=sys.stderr)
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    """Build a cartridge from a YAML file or template."""
    if args.template:
        try:
            builder = build_from_template(args.template, args.name or "custom")
            if args.output:
                cart = builder.build()
                path = package_cartridge(cart, args.output)
                print(f"Built and packaged: {path}")
            else:
                print(builder.to_yaml())
            return
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Build from YAML file
    source = args.source
    if not os.path.exists(source):
        print(f"File not found: {source}", file=sys.stderr)
        sys.exit(1)
    try:
        cart = load_cartridge_from_file(source)
        reg = _ensure_registry()
        reg.register(cart)
        print(f"Built and registered cartridge: {cart.name} v{cart.version}")
        print(f"  Tools: {', '.join(cart.tools.keys())}")
        print(f"  Capabilities: {', '.join(cart.metadata.capabilities) or '—'}")
        if args.output:
            path = package_cartridge(cart, args.output)
            print(f"  Packaged: {path}")
    except Exception as e:
        print(f"Build error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_test(args: argparse.Namespace) -> None:
    """Test a cartridge."""
    reg = _ensure_registry()
    cart = reg.get(args.name)
    if cart is None:
        print(f"Cartridge not found: {args.name}", file=sys.stderr)
        sys.exit(1)
    result = run_cartridge_tests(cart)
    print(result.summary())
    for t in result.tests:
        icon = "✓" if t.passed else "✕"
        detail = f" ({t.error})" if t.error else ""
        print(f"  {icon} {t.test_name}{detail}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start cartridge service (basic REPL)."""
    reg = _ensure_registry()
    bridge = _ensure_bridge()
    scene_mgr = _ensure_scene_manager()

    host = args.host or "0.0.0.0"
    port = args.port or 8471
    print(f"Cartridge Agent Service starting on {host}:{port}")
    print(f"  Cartridges: {len(reg.list_all())}")
    print(f"  Active: {len(reg.list_active())}")
    print(f"  Scenes: {len(scene_mgr._scenes)}")
    print("Press Ctrl+C to stop.\n")

    # Simple status loop — in production this would be a real server
    try:
        import time
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nService stopped.")


def cmd_scene_create(args: argparse.Namespace) -> None:
    """Create a scene from cartridges."""
    mgr = _ensure_scene_manager()
    cart_names = [c.strip() for c in args.cartridges.split(",") if c.strip()]
    scene = mgr.create_scene(
        room_id=args.name,
        cartridge_names=cart_names,
        skin_name=args.skin,
        model_id=args.model,
        schedule=args.schedule or "always",
    )
    if scene is None:
        print("Failed to create scene — check cartridge names exist.", file=sys.stderr)
        sys.exit(1)
    print(f"Created scene: {scene.scene_id}")
    print(f"  Room: {scene.room_id}")
    print(f"  Cartridges: {', '.join(scene.cartridge_names)}")
    print(f"  Schedule: {scene.schedule}")


def cmd_scene_switch(args: argparse.Namespace) -> None:
    """Switch the active scene."""
    mgr = _ensure_scene_manager()
    scene = mgr.get_scene_by_name(args.name)
    if scene is None:
        scene = mgr.get_scene(args.name)
    if scene is None:
        print(f"Scene not found: {args.name}", file=sys.stderr)
        sys.exit(1)
    ok = mgr.activate_scene(scene.scene_id, reason="CLI switch")
    if ok:
        print(f"Active scene: {scene.scene_id}")
        print(f"  Cartridges: {', '.join(scene.cartridge_names)}")
    else:
        print("Failed to activate scene.", file=sys.stderr)
        sys.exit(1)


def cmd_scene_list(args: argparse.Namespace) -> None:
    """List all scenes."""
    mgr = _ensure_scene_manager()
    scenes = mgr._scenes.values()
    if not scenes:
        print("No scenes created.")
        return
    rows = []
    for s in scenes:
        active = "●" if s.active else "○"
        rows.append([
            active,
            s.room_id,
            ", ".join(s.cartridge_names),
            s.model_id,
            s.schedule,
        ])
    _print_table(["", "Name", "Cartridges", "Model", "Schedule"], rows)


def cmd_onboard(args: argparse.Namespace) -> None:
    """Set up the agent — create config directory and load defaults."""
    config_dir = _get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    print(f"Config directory: {config_dir}")

    # Load built-in cartridges
    reg = _ensure_registry()
    print(f"Built-in cartridges: {len(reg.list_all())}")

    # Create default rooms
    bridge = _ensure_bridge()
    defaults = [
        ("nav", "Navigation Bay", "navigation", "field-commander", "glm-5.1"),
        ("engineering", "Engineering", "spreader-loop", "rival", "deepseek-chat"),
        ("bridge", "Bridge", "oracle-relay", "c3po", "glm-5-turbo"),
        ("guardian", "Guardian Station", "fleet-guardian", "straight-man", "glm-4.7"),
    ]
    for room_id, name, cart, skin, model in defaults:
        bridge.create_room(room_id, name, cart, skin, model)

    print(f"Default rooms created: {len(bridge.rooms)}")
    print("Onboard complete. Run 'status' to see agent state.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show agent status."""
    reg = _ensure_registry()
    bridge = _ensure_bridge()
    mgr = _ensure_scene_manager()

    print("╔════════════════════════════════════════╗")
    print("║  Cartridge Agent — Status             ║")
    print("╚════════════════════════════════════════╝")
    print()

    bridge_status = bridge.status()
    scene_status = mgr.status()

    print(f"Cartridges:    {bridge_status['cartridges']} total, "
          f"{bridge_status['active_cartridges']} active")
    print(f"Rooms:         {bridge_status['rooms']} total, "
          f"{bridge_status['active_rooms']} active")
    print(f"Skins:         {bridge_status['skins']}")
    print(f"Scenes:        {scene_status['total_scenes']} total")
    print(f"Active scene:  {scene_status['active_scene'] or '(none)'}")
    print(f"Transitions:   {scene_status['transitions_count']}")
    print()

    if args.verbose:
        carts = reg.list_all()
        if carts:
            print("Cartridges:")
            for c in carts:
                print(f"  {c.status_line()}")
            print()

        rooms = bridge.rooms.values()
        if rooms:
            print("Rooms:")
            for r in rooms:
                active = bridge.is_room_active(r.room_id)
                icon = "●" if active else "○"
                print(f"  {icon} {r.room_id} → {r.cartridge_name or '(none)'} "
                      f"[{r.schedule}]")
            print()

    print("Config dir:", _get_config_dir())


# ======================================================================
# Argument Parser
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cartridge-agent",
        description="Cartridge Agent — Loadable Capability Modules",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start cartridge service")
    p_serve.add_argument("--host", default=None, help="Bind host")
    p_serve.add_argument("--port", type=int, default=None, help="Bind port")

    # list
    sub.add_parser("list", help="List available cartridges")

    # load
    p_load = sub.add_parser("load", help="Load a cartridge")
    p_load.add_argument("name", help="Cartridge name")

    # unload
    p_unload = sub.add_parser("unload", help="Unload a cartridge")
    p_unload.add_argument("name", help="Cartridge name")

    # build
    p_build = sub.add_parser("build", help="Build a cartridge from source")
    p_build.add_argument("source", nargs="?", help="YAML source file")
    p_build.add_argument("--name", help="Cartridge name (for templates)")
    p_build.add_argument("--template", help="Build from template", choices=TEMPLATE_NAMES)
    p_build.add_argument("--output", "-o", help="Output directory for package")

    # test
    p_test = sub.add_parser("test", help="Test a cartridge")
    p_test.add_argument("name", help="Cartridge name")

    # scene
    p_scene = sub.add_parser("scene", help="Scene management")
    scene_sub = p_scene.add_subparsers(dest="scene_command")

    p_sc = scene_sub.add_parser("create", help="Create a scene")
    p_sc.add_argument("--name", required=True, help="Scene/room name")
    p_sc.add_argument("--cartridges", required=True,
                      help="Comma-separated cartridge names")
    p_sc.add_argument("--skin", default=None, help="Skin name")
    p_sc.add_argument("--model", default=None, help="Model override")
    p_sc.add_argument("--schedule", default=None, help="Schedule")

    p_ss = scene_sub.add_parser("switch", help="Switch active scene")
    p_ss.add_argument("name", help="Scene name or ID")

    scene_sub.add_parser("list", help="List scenes")

    # onboard
    sub.add_parser("onboard", help="Set up the agent")

    # status
    p_status = sub.add_parser("status", help="Show agent status")
    p_status.add_argument("-v", "--verbose", action="store_true",
                          help="Show detailed status")

    return parser


# ======================================================================
# Entry Point
# ======================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "load":
        cmd_load(args)
    elif args.command == "unload":
        cmd_unload(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "scene":
        if getattr(args, "scene_command", None) == "create":
            cmd_scene_create(args)
        elif getattr(args, "scene_command", None) == "switch":
            cmd_scene_switch(args)
        elif getattr(args, "scene_command", None) == "list":
            cmd_scene_list(args)
        else:
            parser.parse_args(["scene", "--help"])
    elif args.command == "onboard":
        cmd_onboard(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
