"""
Cartridge Agent — Loadable Capability Modules

Core cartridge system extracted from FLUX-LCAR core.
A cartridge IS a swappable behavior module:
  ROOM × CARTRIDGE × SKIN × MODEL × TIME

Lifecycle: load → activate → execute → deactivate → unload
Supports hot-swap (replace cartridge without restarting).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ======================================================================
# Enums
# ======================================================================

class CartridgeState(Enum):
    """Lifecycle states for a cartridge."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"


class Formality(Enum):
    """MUD formality modes mapped from cartridge skins."""
    NAVAL = "NAVAL"
    PROFESSIONAL = "PROFESSIONAL"
    TNG = "TNG"
    CASUAL = "CASUAL"
    MINIMAL = "MINIMAL"


# ======================================================================
# Cartridge Metadata
# ======================================================================

@dataclass
class CartridgeMetadata:
    """Immutable metadata for a cartridge."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    model_preference: str = "glm-5-turbo"
    compatible_models: List[str] = field(default_factory=list)
    trust_threshold: float = 0.0
    git_repo: str = ""
    onboarding_human: str = ""
    onboarding_agent: str = ""
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute a SHA-256 checksum over the metadata."""
        raw = json.dumps(
            {k: v for k, v in self.__dict__.items() if k != "checksum"},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "model_preference": self.model_preference,
            "compatible_models": self.compatible_models,
            "trust_threshold": self.trust_threshold,
            "git_repo": self.git_repo,
            "onboarding_human": self.onboarding_human,
            "onboarding_agent": self.onboarding_agent,
            "checksum": self.checksum,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CartridgeMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ======================================================================
# Cartridge Tool Definition
# ======================================================================

@dataclass
class CartridgeTool:
    """A single tool/command exposed by a cartridge."""
    name: str
    description: str = ""
    handler: Optional[Callable[..., Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required_capabilities": self.required_capabilities,
        }


# ======================================================================
# Cartridge — Loadable Capability Module
# ======================================================================

class Cartridge:
    """A swappable behavior cartridge with full lifecycle management.

    Lifecycle:
        UNLOADED → load() → LOADED → activate() → ACTIVE
        ACTIVE → deactivate() → LOADED
        LOADED → unload() → UNLOADED
        Any state → error() → ERROR
    """

    def __init__(
        self,
        metadata: CartridgeMetadata,
        tools: Optional[List[CartridgeTool]] = None,
    ) -> None:
        self.metadata = metadata
        self._tools: Dict[str, CartridgeTool] = {}
        self._state: CartridgeState = CartridgeState.UNLOADED
        self._error: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._activated_at: Optional[float] = None
        self._execution_count: int = 0

        if tools:
            for tool in tools:
                self._tools[tool.name] = tool

        # Compute checksum on init if not set
        if not self.metadata.checksum:
            self.metadata.checksum = self.metadata.compute_checksum()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def state(self) -> CartridgeState:
        return self._state

    @property
    def tools(self) -> Dict[str, CartridgeTool]:
        return dict(self._tools)

    @property
    def is_active(self) -> bool:
        return self._state == CartridgeState.ACTIVE

    @property
    def is_loaded(self) -> bool:
        return self._state in (CartridgeState.LOADED, CartridgeState.ACTIVE)

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Transition from UNLOADED → LOADED."""
        if self._state == CartridgeState.LOADED:
            return
        if self._state == CartridgeState.ACTIVE:
            raise RuntimeError(f"Cannot load active cartridge '{self.name}'")
        self._state = CartridgeState.LOADED
        self._loaded_at = time.time()
        self._error = None

    def activate(self) -> None:
        """Transition from LOADED → ACTIVE."""
        if self._state == CartridgeState.ACTIVE:
            return
        if self._state != CartridgeState.LOADED:
            raise RuntimeError(
                f"Cannot activate cartridge '{self.name}' in state {self._state.value}"
            )
        self._state = CartridgeState.ACTIVE
        self._activated_at = time.time()

    def execute(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a command on this cartridge. Requires ACTIVE state."""
        if self._state != CartridgeState.ACTIVE:
            raise RuntimeError(
                f"Cannot execute on cartridge '{self.name}' in state {self._state.value}"
            )
        tool = self._tools.get(command)
        if tool is None:
            raise KeyError(f"Unknown command '{command}' in cartridge '{self.name}'")
        if tool.handler is None:
            raise RuntimeError(f"No handler for command '{command}'")
        self._execution_count += 1
        return tool.handler(*args, **kwargs)

    def deactivate(self) -> None:
        """Transition from ACTIVE → LOADED."""
        if self._state != CartridgeState.ACTIVE:
            raise RuntimeError(
                f"Cannot deactivate cartridge '{self.name}' in state {self._state.value}"
            )
        self._state = CartridgeState.LOADED
        self._activated_at = None

    def unload(self) -> None:
        """Transition from LOADED → UNLOADED."""
        if self._state == CartridgeState.ACTIVE:
            self.deactivate()
        if self._state != CartridgeState.LOADED:
            raise RuntimeError(
                f"Cannot unload cartridge '{self.name}' in state {self._state.value}"
            )
        self._state = CartridgeState.UNLOADED
        self._loaded_at = None

    def set_error(self, msg: str) -> None:
        """Transition to ERROR state."""
        self._error = msg
        self._state = CartridgeState.ERROR

    # ------------------------------------------------------------------
    # Hot-swap
    # ------------------------------------------------------------------

    def hot_swap(self, new_metadata: CartridgeMetadata) -> bool:
        """Replace cartridge metadata without changing state.

        Returns True if the swap was performed, False if checksums match.
        """
        old_checksum = self.metadata.checksum
        new_checksum = new_metadata.compute_checksum()
        if old_checksum == new_checksum:
            return False
        was_active = self._state == CartridgeState.ACTIVE
        self.unload()
        self.metadata = new_metadata
        self.metadata.checksum = new_checksum
        self.load()
        if was_active:
            self.activate()
        return True

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: CartridgeTool) -> None:
        """Register a tool on this cartridge."""
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        """Remove a tool from this cartridge."""
        self._tools.pop(name, None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "state": self._state.value,
            "tools": [t.to_dict() for t in self._tools.values()],
            "loaded_at": self._loaded_at,
            "activated_at": self._activated_at,
            "execution_count": self._execution_count,
            "error": self._error,
        }

    def status_line(self) -> str:
        """One-line status summary."""
        state_icon = {
            CartridgeState.UNLOADED: "○",
            CartridgeState.LOADED: "◐",
            CartridgeState.ACTIVE: "●",
            CartridgeState.ERROR: "✕",
        }.get(self._state, "?")
        return f"{state_icon} {self.name} v{self.version} [{self._state.value}]"


# ======================================================================
# CartridgeRegistry — Manage Available Cartridges
# ======================================================================

class CartridgeRegistry:
    """Central registry for all cartridges with dependency resolution."""

    def __init__(self) -> None:
        self._cartridges: Dict[str, Cartridge] = {}
        self._event_hooks: Dict[str, List[Callable[[Cartridge], None]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cartridge: Cartridge) -> None:
        """Register a cartridge. Replaces existing with same name."""
        old = self._cartridges.get(cartridge.name)
        self._cartridges[cartridge.name] = cartridge
        if old:
            self._fire_event("replaced", cartridge)
        else:
            self._fire_event("registered", cartridge)

    def unregister(self, name: str) -> Optional[Cartridge]:
        """Remove and return a cartridge."""
        cart = self._cartridges.pop(name, None)
        if cart:
            self._fire_event("unregistered", cart)
        return cart

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Cartridge]:
        return self._cartridges.get(name)

    def list_all(self) -> List[Cartridge]:
        return list(self._cartridges.values())

    def list_active(self) -> List[Cartridge]:
        return [c for c in self._cartridges.values() if c.is_active]

    def list_by_capability(self, capability: str) -> List[Cartridge]:
        return [
            c for c in self._cartridges.values()
            if capability in c.metadata.capabilities
        ]

    def list_dict(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._cartridges.values()]

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def load(self, name: str) -> bool:
        cart = self._cartridges.get(name)
        if cart is None:
            return False
        try:
            cart.load()
            self._fire_event("loaded", cart)
            return True
        except RuntimeError:
            return False

    def unload(self, name: str) -> bool:
        cart = self._cartridges.get(name)
        if cart is None:
            return False
        try:
            cart.unload()
            self._fire_event("unloaded", cart)
            return True
        except RuntimeError:
            return False

    def activate(self, name: str) -> bool:
        cart = self._cartridges.get(name)
        if cart is None:
            return False
        try:
            cart.activate()
            self._fire_event("activated", cart)
            return True
        except RuntimeError:
            return False

    def deactivate(self, name: str) -> bool:
        cart = self._cartridges.get(name)
        if cart is None:
            return False
        try:
            cart.deactivate()
            self._fire_event("deactivated", cart)
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Hot-swap
    # ------------------------------------------------------------------

    def hot_swap(self, name: str, new_metadata: CartridgeMetadata) -> bool:
        cart = self._cartridges.get(name)
        if cart is None:
            return False
        swapped = cart.hot_swap(new_metadata)
        if swapped:
            self._fire_event("hot_swapped", cart)
        return swapped

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def resolve_dependencies(self, name: str) -> List[str]:
        """Return ordered list of dependency names for a cartridge."""
        cart = self._cartridges.get(name)
        if cart is None:
            return []
        result: List[str] = []
        visited: Set[str] = set()

        def _visit(dep_name: str) -> None:
            if dep_name in visited:
                return
            visited.add(dep_name)
            dep_cart = self._cartridges.get(dep_name)
            if dep_cart:
                for d in dep_cart.metadata.dependencies:
                    _visit(d)
            result.append(dep_name)

        for dep in cart.metadata.dependencies:
            _visit(dep)
        return result

    def check_dependencies(self, name: str) -> List[str]:
        """Return list of missing dependencies."""
        cart = self._cartridges.get(name)
        if cart is None:
            return []
        missing: List[str] = []
        for dep in cart.metadata.dependencies:
            if dep not in self._cartridges:
                missing.append(dep)
        return missing

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable[[Cartridge], None]) -> None:
        self._event_hooks.setdefault(event, []).append(handler)

    def _fire_event(self, event: str, cartridge: Cartridge) -> None:
        for handler in self._event_hooks.get(event, []):
            try:
                handler(cartridge)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cartridges": {n: c.to_dict() for n, c in self._cartridges.items()},
            "count": len(self._cartridges),
            "active_count": len(self.list_active()),
        }


# ======================================================================
# Built-in defaults
# ======================================================================

BUILTIN_CARTRIDGES: List[Cartridge] = []

_spreader = Cartridge(CartridgeMetadata(
    name="spreader-loop",
    description="Modify-spread-tool-reflect iterative engine",
    capabilities=["spreader"],
    onboarding_human="I modify, spread, verify, and log — then the Reasoner reflects.",
    onboarding_agent="Spreader Loop loaded. Ready for iterative cycles.",
    model_preference="glm-5-turbo",
))
_spreader.add_tool(CartridgeTool("spreader_run", "Execute iteration cycle"))
_spreader.add_tool(CartridgeTool("spreader_status", "Get loop statistics"))
_spreader.add_tool(CartridgeTool("spreader_reflect", "Generate reflection prompt"))
_spreader.add_tool(CartridgeTool("spreader_discover_tiles", "Find new tile patterns"))
_spreader.load()
BUILTIN_CARTRIDGES.append(_spreader)

_oracle = Cartridge(CartridgeMetadata(
    name="oracle-relay",
    description="Iron-to-iron bottle protocol for async fleet communication",
    capabilities=["bottle"],
    onboarding_human="I pass bottles between vessels — no intermediaries needed.",
    onboarding_agent="Oracle Relay active. Bottle protocol ready.",
    model_preference="glm-5-turbo",
))
_oracle.add_tool(CartridgeTool("bottle_send", "Send bottle to vessel"))
_oracle.add_tool(CartridgeTool("bottle_read", "Read bottles addressed to us"))
_oracle.add_tool(CartridgeTool("bottle_list", "List pending bottles"))
_oracle.add_tool(CartridgeTool("bottle_reply", "Reply to a bottle"))
_oracle.load()
BUILTIN_CARTRIDGES.append(_oracle)

_guardian = Cartridge(CartridgeMetadata(
    name="fleet-guardian",
    description="External watchdog for agent runtimes",
    capabilities=["guardian"],
    onboarding_human="I monitor vessel health and enforce timeouts.",
    onboarding_agent="Fleet Guardian on watch. Monitoring active.",
    model_preference="glm-4.7",
))
_guardian.add_tool(CartridgeTool("health_check", "Check vessel health"))
_guardian.add_tool(CartridgeTool("stuck_detect", "Detect stuck states"))
_guardian.add_tool(CartridgeTool("timeout_enforce", "Enforce execution timeout"))
_guardian.load()
BUILTIN_CARTRIDGES.append(_guardian)

_nav = Cartridge(CartridgeMetadata(
    name="navigation",
    description="Real-time navigation with ESP32 sensor feeds",
    capabilities=["navigation"],
    onboarding_human="I hold course, read sensors, and adjust rudder automatically.",
    onboarding_agent="Navigation cartridge loaded. Sensors online.",
    model_preference="glm-5.1",
    trust_threshold=0.5,
    git_repo="Lucineer/holodeck-c",
))
_nav.add_tool(CartridgeTool("read_compass", "Get current heading"))
_nav.add_tool(CartridgeTool("set_course", "Set target heading"))
_nav.add_tool(CartridgeTool("adjust_rudder", "Adjust rudder angle"))
_nav.add_tool(CartridgeTool("check_depth", "Read depth sounder"))
_nav.load()
BUILTIN_CARTRIDGES.append(_nav)
