"""
Cartridge Agent — Scene Manager

Manages composed scenes from multiple cartridges.
A Scene = ROOM × CARTRIDGE × SKIN × MODEL × TIME

Responsibilities:
- Scene composition from multiple cartridges
- Scene transitions
- Scene state management
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from cartridge import (
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeState,
    Formality,
)


# ======================================================================
# Scene
# ======================================================================

@dataclass
class Scene:
    """Complete agent interaction context.

    Every dimension of a running agent session lives here:
    where it is (room), what it does (cartridge), how it speaks (skin),
    what thinks for it (model), and when it's allowed (schedule).
    """

    scene_id: str = ""
    room_id: str = ""
    cartridge_names: List[str] = field(default_factory=list)
    skin_name: Optional[str] = None
    model_id: str = "glm-5-turbo"
    schedule: str = "always"
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    trust_required: float = 0.0
    created_at: float = field(default_factory=time.time)
    active: bool = False

    def __post_init__(self) -> None:
        if not self.scene_id:
            cart_key = ",".join(sorted(self.cartridge_names)) or "empty"
            self.scene_id = f"{self.room_id}:{cart_key}:{id(self):08x}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "room_id": self.room_id,
            "cartridge_names": self.cartridge_names,
            "skin_name": self.skin_name,
            "model_id": self.model_id,
            "schedule": self.schedule,
            "context": self.context,
            "priority": self.priority,
            "trust_required": self.trust_required,
            "created_at": self.created_at,
            "active": self.active,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scene:
        return cls(
            scene_id=data.get("scene_id", ""),
            room_id=data.get("room_id", ""),
            cartridge_names=data.get("cartridge_names", []),
            skin_name=data.get("skin_name"),
            model_id=data.get("model_id", "glm-5-turbo"),
            schedule=data.get("schedule", "always"),
            context=data.get("context", {}),
            priority=data.get("priority", 0),
            trust_required=data.get("trust_required", 0.0),
            created_at=data.get("created_at", time.time()),
            active=data.get("active", False),
        )

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(self.scene_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scene):
            return self.scene_id == other.scene_id
        return NotImplemented

    def clone(self, **overrides: Any) -> Scene:
        """Return a copy with fields overridden."""
        d = self.to_dict()
        d.pop("scene_id", None)
        d.update(overrides)
        return Scene.from_dict(d)


# ======================================================================
# SceneTransition
# ======================================================================

@dataclass
class SceneTransition:
    """Record of a scene transition event."""
    from_scene: Optional[str]
    to_scene: str
    timestamp: float = field(default_factory=time.time)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_scene": self.from_scene,
            "to_scene": self.to_scene,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


# ======================================================================
# SceneManager
# ======================================================================

class SceneManager:
    """Compose and manage scenes from cartridges.

    Scenes aggregate one or more cartridges into a unified
    interaction context for a MUD room.
    """

    def __init__(self, registry: Optional[CartridgeRegistry] = None) -> None:
        self.registry = registry or CartridgeRegistry()
        self._scenes: Dict[str, Scene] = {}
        self._active_scene_id: Optional[str] = None
        self._transitions: List[SceneTransition] = []
        self._schedule_map: Dict[str, List[str]] = {}  # schedule -> scene_ids

    # ------------------------------------------------------------------
    # Scene CRUD
    # ------------------------------------------------------------------

    def create_scene(
        self,
        room_id: str,
        cartridge_names: List[str],
        skin_name: Optional[str] = None,
        model_id: Optional[str] = None,
        schedule: str = "always",
        priority: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Scene]:
        """Create a new scene from cartridge names.

        Returns None if any cartridge is not found in the registry.
        """
        # Verify all cartridges exist
        for name in cartridge_names:
            cart = self.registry.get(name)
            if cart is None:
                return None

        effective_model = model_id or "glm-5-turbo"
        # Use preferred model of the first cartridge if not specified
        if not model_id and cartridge_names:
            first = self.registry.get(cartridge_names[0])
            if first:
                effective_model = first.metadata.model_preference

        scene = Scene(
            room_id=room_id,
            cartridge_names=list(cartridge_names),
            skin_name=skin_name,
            model_id=effective_model,
            schedule=schedule,
            priority=priority,
            context=context or {},
        )
        self._scenes[scene.scene_id] = scene

        # Update schedule index
        self._schedule_map.setdefault(schedule, []).append(scene.scene_id)

        return scene

    def remove_scene(self, scene_id: str) -> Optional[Scene]:
        scene = self._scenes.pop(scene_id, None)
        if scene:
            if self._active_scene_id == scene_id:
                self._active_scene_id = None
            # Clean schedule map
            for sched in self._schedule_map.get(scene.schedule, []):
                if sched == scene_id:
                    self._schedule_map[scene.schedule].remove(scene_id)
        return scene

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        return self._scenes.get(scene_id)

    def get_scene_by_name(self, name: str) -> Optional[Scene]:
        """Find a scene by room_id (used as name in CLI)."""
        for scene in self._scenes.values():
            if scene.room_id == name:
                return scene
        return None

    def list_scenes(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._scenes.values()]

    # ------------------------------------------------------------------
    # Activation & Transitions
    # ------------------------------------------------------------------

    def activate_scene(self, scene_id: str, reason: str = "") -> bool:
        """Switch active scene, recording a transition."""
        scene = self._scenes.get(scene_id)
        if scene is None:
            return False

        # Load and activate all cartridge dependencies
        for cart_name in scene.cartridge_names:
            self.registry.load(cart_name)
            self.registry.activate(cart_name)

        # Record transition
        self._transitions.append(SceneTransition(
            from_scene=self._active_scene_id,
            to_scene=scene_id,
            reason=reason,
        ))

        # Deactivate previous scene's cartridges
        if self._active_scene_id and self._active_scene_id != scene_id:
            prev = self._scenes.get(self._active_scene_id)
            if prev:
                for cart_name in prev.cartridge_names:
                    self.registry.deactivate(cart_name)

        self._active_scene_id = scene_id
        scene.active = True
        return True

    def deactivate_active(self, reason: str = "") -> bool:
        """Deactivate the current active scene."""
        if self._active_scene_id is None:
            return False
        scene = self._scenes.get(self._active_scene_id)
        if scene is None:
            return False
        for cart_name in scene.cartridge_names:
            self.registry.deactivate(cart_name)
        self._transitions.append(SceneTransition(
            from_scene=self._active_scene_id,
            to_scene="none",
            reason=reason,
        ))
        scene.active = False
        self._active_scene_id = None
        return True

    @property
    def active_scene(self) -> Optional[Scene]:
        if self._active_scene_id is None:
            return None
        return self._scenes.get(self._active_scene_id)

    @property
    def transitions(self) -> List[SceneTransition]:
        return list(self._transitions)

    # ------------------------------------------------------------------
    # Schedule-based activation
    # ------------------------------------------------------------------

    def best_scene_for_room(self, room_id: str) -> Optional[Scene]:
        """Pick the best scene for a room based on schedule and priority."""
        candidates = [
            s for s in self._scenes.values()
            if s.room_id == room_id
        ]
        if not candidates:
            return None

        hour = datetime.datetime.now().hour
        valid: List[Scene] = []
        for scene in candidates:
            if scene.schedule == "always":
                valid.append(scene)
            elif scene.schedule == "nighttime" and (hour < 6 or hour >= 22):
                valid.append(scene)
            elif scene.schedule == "daytime" and 6 <= hour < 22:
                valid.append(scene)

        if not valid:
            valid = candidates  # fallback

        return max(valid, key=lambda s: s.priority)

    # ------------------------------------------------------------------
    # Scene state
    # ------------------------------------------------------------------

    def update_scene_context(self, scene_id: str, **kwargs: Any) -> bool:
        scene = self._scenes.get(scene_id)
        if scene is None:
            return False
        scene.context.update(kwargs)
        return True

    def get_transitions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._transitions[-limit:]]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "total_scenes": len(self._scenes),
            "active_scene": self._active_scene_id,
            "transitions_count": len(self._transitions),
            "cartridges_in_use": self._count_active_cartridges(),
        }

    def _count_active_cartridges(self) -> int:
        names: Set[str] = set()
        if self._active_scene_id:
            scene = self._scenes.get(self._active_scene_id)
            if scene:
                names.update(scene.cartridge_names)
        return len(names)
