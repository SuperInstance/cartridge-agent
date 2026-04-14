"""
Cartridge Agent — Cartridge×MUD Bridge

Connects the cartridge system to a MUD room architecture.
A cartridge IS a MUD room configuration:
  ROOM × CARTRIDGE × SKIN × MODEL × TIME

Responsibilities:
- Bridge cartridges to the MUD room system
- Room-cartridge mapping
- Command routing through cartridges
- Skin/personality integration with MUD formality modes
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cartridge import (
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    Formality,
    CartridgeState,
)


# ======================================================================
# Skin Definition
# ======================================================================

@dataclass
class Skin:
    """A personality skin — maps to MUD formality + personality."""
    name: str
    description: str = ""
    formality: Formality = Formality.TNG
    system_prompt_suffix: str = ""
    temperature: float = 0.7
    tool_preferences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "formality": self.formality.value,
            "system_prompt_suffix": self.system_prompt_suffix,
            "temperature": self.temperature,
            "tool_preferences": self.tool_preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Skin:
        data = dict(data)
        if "formality" in data and isinstance(data["formality"], str):
            data["formality"] = Formality(data["formality"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ======================================================================
# Room Definition
# ======================================================================

@dataclass
class Room:
    """A MUD room that can host a cartridge."""
    room_id: str
    name: str = ""
    description: str = ""
    cartridge_name: Optional[str] = None
    skin_name: Optional[str] = None
    model: str = "glm-5-turbo"
    schedule: str = "always"
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "description": self.description,
            "cartridge_name": self.cartridge_name,
            "skin_name": self.skin_name,
            "model": self.model,
            "schedule": self.schedule,
            "priority": self.priority,
        }


# ======================================================================
# Command Context
# ======================================================================

@dataclass
class CommandContext:
    """Context for a routed command."""
    room_id: str
    command: str
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    source: str = "user"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            import time
            self.timestamp = time.time()


# ======================================================================
# Command Result
# ======================================================================

@dataclass
class CommandResult:
    """Result of a routed command."""
    success: bool
    output: Any = None
    error: str = ""
    command: str = ""
    room_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "command": self.command,
            "room_id": self.room_id,
        }


# ======================================================================
# CartridgeBridge
# ======================================================================

class CartridgeBridge:
    """Bridges cartridges to the MUD room system.

    Manages room-cartridge mappings and routes commands
    through the appropriate cartridge's tools.
    """

    def __init__(
        self,
        registry: Optional[CartridgeRegistry] = None,
    ) -> None:
        self.registry = registry or CartridgeRegistry()
        self.rooms: Dict[str, Room] = {}
        self.skins: Dict[str, Skin] = {}
        self._command_interceptors: List[
            Callable[[CommandContext], Optional[CommandResult]]
        ] = []
        self._register_default_skins()

    # ------------------------------------------------------------------
    # Skin management
    # ------------------------------------------------------------------

    def _register_default_skins(self) -> None:
        defaults = [
            Skin("straight-man", "Abbott & Costello straight man",
                 Formality.PROFESSIONAL),
            Skin("funny-man", "Abbott & Costello funny man",
                 Formality.CASUAL, temperature=0.9),
            Skin("penn", "Penn (explainer)", Formality.TNG, temperature=0.7),
            Skin("teller", "Teller (silent doer)", Formality.MINIMAL,
                 temperature=0.3),
            Skin("r2d2", "R2-D2 (beeps and whistles)", Formality.CASUAL,
                 temperature=0.8),
            Skin("c3po", "C-3PO (formal protocol)", Formality.NAVAL,
                 temperature=0.4),
            Skin("rival", "Competitive rival", Formality.TNG, temperature=0.85),
            Skin("field-commander", "Military field commander", Formality.NAVAL,
                 temperature=0.5),
        ]
        for skin in defaults:
            self.skins[skin.name] = skin

    def register_skin(self, skin: Skin) -> None:
        self.skins[skin.name] = skin

    def unregister_skin(self, name: str) -> None:
        self.skins.pop(name, None)

    def get_skin(self, name: str) -> Optional[Skin]:
        return self.skins.get(name)

    def list_skins(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.skins.values()]

    # ------------------------------------------------------------------
    # Room management
    # ------------------------------------------------------------------

    def create_room(
        self,
        room_id: str,
        name: str = "",
        cartridge_name: Optional[str] = None,
        skin_name: Optional[str] = None,
        model: str = "glm-5-turbo",
        schedule: str = "always",
        priority: int = 0,
    ) -> Room:
        room = Room(
            room_id=room_id,
            name=name or room_id,
            cartridge_name=cartridge_name,
            skin_name=skin_name,
            model=model,
            schedule=schedule,
            priority=priority,
        )
        self.rooms[room_id] = room
        return room

    def remove_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.pop(room_id, None)

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def list_rooms(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.rooms.values()]

    # ------------------------------------------------------------------
    # Room-cartridge mapping
    # ------------------------------------------------------------------

    def assign_cartridge(self, room_id: str, cartridge_name: str) -> bool:
        room = self.rooms.get(room_id)
        if room is None:
            return False
        cart = self.registry.get(cartridge_name)
        if cart is None:
            return False
        room.cartridge_name = cartridge_name
        # Auto-load the cartridge if not loaded
        if not cart.is_loaded:
            self.registry.load(cartridge_name)
        return True

    def assign_skin(self, room_id: str, skin_name: str) -> bool:
        room = self.rooms.get(room_id)
        if room is None:
            return False
        if skin_name not in self.skins:
            return False
        room.skin_name = skin_name
        return True

    def is_room_active(self, room_id: str) -> bool:
        """Check if a room's cartridge schedule is currently valid."""
        room = self.rooms.get(room_id)
        if room is None or room.cartridge_name is None:
            return False
        cart = self.registry.get(room.cartridge_name)
        if cart is None or not cart.is_active:
            return False
        return self._schedule_active(room.schedule)

    @staticmethod
    def _schedule_active(schedule: str) -> bool:
        """Check if a schedule is active at the current time."""
        if schedule == "always":
            return True
        hour = datetime.datetime.now().hour
        if schedule == "nighttime" and (hour < 6 or hour >= 22):
            return True
        if schedule == "daytime" and 6 <= hour < 22:
            return True
        return False

    # ------------------------------------------------------------------
    # Command routing
    # ------------------------------------------------------------------

    def add_interceptor(
        self,
        interceptor: Callable[[CommandContext], Optional[CommandResult]],
    ) -> None:
        self._command_interceptors.append(interceptor)

    def route_command(self, ctx: CommandContext) -> CommandResult:
        """Route a command through interceptors → room cartridge."""
        # Run interceptors first
        for interceptor in self._command_interceptors:
            result = interceptor(ctx)
            if result is not None:
                return result

        room = self.rooms.get(ctx.room_id)
        if room is None:
            return CommandResult(
                success=False, error=f"Unknown room '{ctx.room_id}'",
                command=ctx.command, room_id=ctx.room_id,
            )

        if room.cartridge_name is None:
            return CommandResult(
                success=False, error="No cartridge assigned to room",
                command=ctx.command, room_id=ctx.room_id,
            )

        cart = self.registry.get(room.cartridge_name)
        if cart is None:
            return CommandResult(
                success=False,
                error=f"Cartridge '{room.cartridge_name}' not found",
                command=ctx.command, room_id=ctx.room_id,
            )

        if not cart.is_active:
            return CommandResult(
                success=False,
                error=f"Cartridge '{room.cartridge_name}' is {cart.state.value}",
                command=ctx.command, room_id=ctx.room_id,
            )

        try:
            output = cart.execute(ctx.command, *ctx.args, **ctx.kwargs)
            return CommandResult(
                success=True, output=output,
                command=ctx.command, room_id=ctx.room_id,
            )
        except KeyError:
            return CommandResult(
                success=False,
                error=f"Unknown command '{ctx.command}' in cartridge '{cart.name}'",
                command=ctx.command, room_id=ctx.room_id,
            )
        except RuntimeError as exc:
            return CommandResult(
                success=False, error=str(exc),
                command=ctx.command, room_id=ctx.room_id,
            )

    # ------------------------------------------------------------------
    # MUD config export
    # ------------------------------------------------------------------

    def get_mud_config(self, room_id: str) -> Dict[str, Any]:
        """Get the full MUD configuration for a room."""
        room = self.rooms.get(room_id)
        if room is None:
            return {}

        cart = self.registry.get(room.cartridge_name) if room.cartridge_name else None
        skin = self.skins.get(room.skin_name) if room.skin_name else None

        return {
            "room_id": room.room_id,
            "name": room.name,
            "cartridge": cart.to_dict() if cart else None,
            "skin": skin.to_dict() if skin else None,
            "model": room.model,
            "schedule": room.schedule,
            "commands": list(cart.tools.keys()) if cart else [],
            "active": self.is_room_active(room_id),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "rooms": len(self.rooms),
            "skins": len(self.skins),
            "cartridges": len(self.registry.list_all()),
            "active_cartridges": len(self.registry.list_active()),
            "active_rooms": sum(
                1 for rid in self.rooms if self.is_room_active(rid)
            ),
        }
