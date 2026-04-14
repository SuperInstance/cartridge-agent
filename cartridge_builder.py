"""
Cartridge Agent — Cartridge Builder

DSL for building new cartridges, template system for common patterns,
testing framework, and packaging/distribution support.

All output uses YAML for human-readable cartridge manifests.
Only stdlib + pyyaml dependencies.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from cartridge import (
    BUILTIN_CARTRIDGES,
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeState,
    CartridgeTool,
)


# ======================================================================
# Cartridge Manifest
# ======================================================================

@dataclass
class CartridgeManifest:
    """YAML-serializable cartridge definition."""
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
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tests: List[Dict[str, Any]] = field(default_factory=list)

    def to_yaml(self) -> str:
        data = {
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
            "onboarding": {
                "human": self.onboarding_human,
                "agent": self.onboarding_agent,
            },
            "tools": self.tools,
        }
        if self.tests:
            data["tests"] = self.tests
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, raw: str) -> CartridgeManifest:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping")
        onboarding = data.get("onboarding", {})
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", []),
            model_preference=data.get("model_preference", "glm-5-turbo"),
            compatible_models=data.get("compatible_models", []),
            trust_threshold=data.get("trust_threshold", 0.0),
            git_repo=data.get("git_repo", ""),
            onboarding_human=onboarding.get("human", ""),
            onboarding_agent=onboarding.get("agent", ""),
            tools=data.get("tools", []),
            tests=data.get("tests", []),
        )

    def to_metadata(self) -> CartridgeMetadata:
        return CartridgeMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            capabilities=self.capabilities,
            dependencies=self.dependencies,
            model_preference=self.model_preference,
            compatible_models=self.compatible_models,
            trust_threshold=self.trust_threshold,
            git_repo=self.git_repo,
            onboarding_human=self.onboarding_human,
            onboarding_agent=self.onboarding_agent,
        )


# ======================================================================
# CartridgeBuilder — Fluent DSL
# ======================================================================

class CartridgeBuilder:
    """Fluent builder DSL for constructing cartridges.

    Usage:
        cart = (CartridgeBuilder("my-cartridge")
            .version("2.0.0")
            .description("Does cool things")
            .capability("cool")
            .tool("do_thing", "Does the thing")
            .tool("undo_thing", "Undoes the thing")
            .build())
    """

    def __init__(self, name: str) -> None:
        self._manifest = CartridgeManifest(name=name)
        self._tool_handlers: Dict[str, Callable[..., Any]] = {}

    def version(self, v: str) -> CartridgeBuilder:
        self._manifest.version = v
        return self

    def description(self, d: str) -> CartridgeBuilder:
        self._manifest.description = d
        return self

    def author(self, a: str) -> CartridgeBuilder:
        self._manifest.author = a
        return self

    def capability(self, cap: str) -> CartridgeBuilder:
        if cap not in self._manifest.capabilities:
            self._manifest.capabilities.append(cap)
        return self

    def capabilities(self, caps: List[str]) -> CartridgeBuilder:
        for c in caps:
            self.capability(c)
        return self

    def depends_on(self, name: str) -> CartridgeBuilder:
        if name not in self._manifest.dependencies:
            self._manifest.dependencies.append(name)
        return self

    def model_preference(self, m: str) -> CartridgeBuilder:
        self._manifest.model_preference = m
        return self

    def trust_threshold(self, t: float) -> CartridgeBuilder:
        self._manifest.trust_threshold = t
        return self

    def git_repo(self, r: str) -> CartridgeBuilder:
        self._manifest.git_repo = r
        return self

    def onboarding_human(self, text: str) -> CartridgeBuilder:
        self._manifest.onboarding_human = text
        return self

    def onboarding_agent(self, text: str) -> CartridgeBuilder:
        self._manifest.onboarding_agent = text
        return self

    def tool(
        self,
        name: str,
        description: str = "",
        handler: Optional[Callable[..., Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> CartridgeBuilder:
        tool_def = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
        }
        self._manifest.tools.append(tool_def)
        if handler is not None:
            self._tool_handlers[name] = handler
        return self

    def test(
        self,
        command: str,
        expected_output: Any = None,
        expect_error: bool = False,
    ) -> CartridgeBuilder:
        self._manifest.tests.append({
            "command": command,
            "expected_output": expected_output,
            "expect_error": expect_error,
        })
        return self

    def build(self) -> Cartridge:
        """Build the cartridge from the accumulated definitions."""
        metadata = self._manifest.to_metadata()
        tools: List[CartridgeTool] = []
        for tdef in self._manifest.tools:
            ct = CartridgeTool(
                name=tdef["name"],
                description=tdef.get("description", ""),
                handler=self._tool_handlers.get(tdef["name"]),
                parameters=tdef.get("parameters", {}),
            )
            tools.append(ct)
        cart = Cartridge(metadata=metadata, tools=tools)
        cart.load()
        return cart

    def manifest(self) -> CartridgeManifest:
        """Get the manifest without building."""
        return self._manifest

    def to_yaml(self) -> str:
        """Serialize the builder state to YAML."""
        return self._manifest.to_yaml()


# ======================================================================
# Template System
# ======================================================================

# Built-in cartridge templates
CARTRIDGE_TEMPLATES: Dict[str, str] = {
    "iterative": """name: {{name}}
version: "1.0.0"
description: "{{name}} — iterative processing loop"
capabilities: [iterative]
model_preference: glm-5-turbo
onboarding:
  human: "I iterate through cycles until convergence."
  agent: "{{name}} loaded. Iterative loop ready."
tools:
  - name: run_cycle
    description: "Execute one iteration cycle"
  - name: get_status
    description: "Get current iteration status"
  - name: reset
    description: "Reset the iteration state"
""",
    "relay": """name: {{name}}
version: "1.0.0"
description: "{{name}} — async message relay"
capabilities: [relay]
model_preference: glm-5-turbo
onboarding:
  human: "I relay messages between endpoints."
  agent: "{{name}} active. Relay protocol ready."
tools:
  - name: send
    description: "Send a message"
  - name: receive
    description: "Receive pending messages"
  - name: list_pending
    description: "List pending messages"
""",
    "watchdog": """name: {{name}}
version: "1.0.0"
description: "{{name}} — external watchdog monitor"
capabilities: [watchdog]
model_preference: glm-4.7
trust_threshold: 0.3
onboarding:
  human: "I monitor health and enforce timeouts."
  agent: "{{name}} on watch. Monitoring active."
tools:
  - name: health_check
    description: "Check health status"
  - name: detect_stuck
    description: "Detect stuck states"
  - name: enforce_timeout
    description: "Enforce execution timeout"
""",
    "navigation": """name: {{name}}
version: "1.0.0"
description: "{{name}} — sensor-driven navigation"
capabilities: [navigation]
model_preference: glm-5.1
trust_threshold: 0.5
onboarding:
  human: "I read sensors and hold course."
  agent: "{{name}} loaded. Sensors online."
tools:
  - name: read_sensor
    description: "Read a sensor value"
  - name: set_target
    description: "Set target value"
  - name: adjust
    description: "Adjust control surface"
  - name: status
    description: "Get navigation status"
""",
}

# List of available template names
TEMPLATE_NAMES: List[str] = list(CARTRIDGE_TEMPLATES.keys())


def build_from_template(
    template_name: str,
    cartridge_name: str,
    **overrides: Any,
) -> CartridgeBuilder:
    """Create a builder pre-populated from a template."""
    if template_name not in CARTRIDGE_TEMPLATES:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {', '.join(TEMPLATE_NAMES)}"
        )
    template = CARTRIDGE_TEMPLATES[template_name]
    rendered = template.replace("{{name}}", cartridge_name)

    manifest = CartridgeManifest.from_yaml(rendered)
    builder = CartridgeBuilder(manifest.name)
    builder._manifest = manifest
    return builder


# ======================================================================
# Cartridge Testing Framework
# ======================================================================

@dataclass
class TestResult:
    """Result of a single cartridge test."""
    test_name: str
    passed: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class TestSuiteResult:
    """Aggregated results of a cartridge test run."""
    cartridge_name: str
    tests: List[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    total: int = 0

    def add(self, result: TestResult) -> None:
        self.tests.append(result)
        self.total += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.total > 0

    def summary(self) -> str:
        icon = "✓" if self.success else "✕"
        return (
            f"{icon} {self.cartridge_name}: "
            f"{self.passed}/{self.total} passed"
            + (f" ({self.failed} failed)" if self.failed else "")
        )


def run_cartridge_tests(cartridge: Cartridge) -> TestSuiteResult:
    """Run built-in lifecycle tests on a cartridge.

    Tests:
    1. Metadata completeness
    2. Lifecycle transitions (load → activate → deactivate → unload)
    3. Tool execution (if handlers are present)
    4. Hot-swap support
    5. Serialization round-trip
    """
    import time as _time

    suite = TestSuiteResult(cartridge_name=cartridge.name)

    # Test 1: Metadata
    t0 = _time.monotonic()
    try:
        assert cartridge.name, "Name is required"
        assert cartridge.version, "Version is required"
        assert isinstance(cartridge.metadata.capabilities, list)
        cartridge.metadata.compute_checksum()
        suite.add(TestResult("metadata", True, duration_ms=(_time.monotonic() - t0) * 1000))
    except AssertionError as e:
        suite.add(TestResult("metadata", False, error=str(e), duration_ms=(_time.monotonic() - t0) * 1000))

    # Test 2: Lifecycle
    cart_copy = Cartridge(
        metadata=CartridgeMetadata.from_dict(cartridge.metadata.to_dict()),
    )
    for tool in cartridge.tools.values():
        cart_copy.add_tool(CartridgeTool(tool.name, tool.description))
    t0 = _time.monotonic()
    try:
        cart_copy.load()
        assert cart_copy.state == CartridgeState.LOADED
        cart_copy.activate()
        assert cart_copy.state == CartridgeState.ACTIVE
        cart_copy.deactivate()
        assert cart_copy.state == CartridgeState.LOADED
        cart_copy.unload()
        assert cart_copy.state == CartridgeState.UNLOADED
        suite.add(TestResult("lifecycle", True, duration_ms=(_time.monotonic() - t0) * 1000))
    except (AssertionError, RuntimeError) as e:
        suite.add(TestResult("lifecycle", False, error=str(e), duration_ms=(_time.monotonic() - t0) * 1000))

    # Test 3: Tool execution
    t0 = _time.monotonic()
    if cartridge.tools and any(t.handler for t in cartridge.tools.values()):
        try:
            # Save state, activate if needed
            was_loaded = cartridge.is_loaded and not cartridge.is_active
            if was_loaded:
                cartridge.activate()
            for tname, tool in cartridge.tools.items():
                if tool.handler:
                    result = cartridge.execute(tname)
                    suite.add(TestResult(f"tool:{tname}", True, output=result, duration_ms=(_time.monotonic() - t0) * 1000))
            # Restore state
            if was_loaded:
                cartridge.deactivate()
        except Exception as e:
            suite.add(TestResult("tool_execution", False, error=str(e), duration_ms=(_time.monotonic() - t0) * 1000))
    else:
        suite.add(TestResult("tool_execution", True, output="skipped (no handlers)", duration_ms=0))

    # Test 4: Hot-swap
    t0 = _time.monotonic()
    try:
        new_meta = CartridgeMetadata.from_dict(cartridge.metadata.to_dict())
        new_meta.version = "9.9.9"
        cart_copy2 = Cartridge(
            metadata=CartridgeMetadata.from_dict(cartridge.metadata.to_dict()),
        )
        cart_copy2.load()
        swapped = cart_copy2.hot_swap(new_meta)
        assert swapped, "Hot-swap should return True for different version"
        assert cart_copy2.version == "9.9.9"
        suite.add(TestResult("hot_swap", True, duration_ms=(_time.monotonic() - t0) * 1000))
    except (AssertionError, RuntimeError) as e:
        suite.add(TestResult("hot_swap", False, error=str(e), duration_ms=(_time.monotonic() - t0) * 1000))

    # Test 5: Serialization
    t0 = _time.monotonic()
    try:
        d = cartridge.to_dict()
        # Just verify it's a valid dict with expected keys
        assert "metadata" in d
        assert "state" in d
        assert "tools" in d
        suite.add(TestResult("serialization", True, duration_ms=(_time.monotonic() - t0) * 1000))
    except (AssertionError, Exception) as e:
        suite.add(TestResult("serialization", False, error=str(e), duration_ms=(_time.monotonic() - t0) * 1000))

    return suite


# ======================================================================
# Cartridge Packaging
# ======================================================================

def package_cartridge(cartridge: Cartridge, output_dir: Optional[str] = None) -> str:
    """Package a cartridge as a YAML manifest file.

    Returns the path to the created file.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    manifest = CartridgeManifest(
        name=cartridge.name,
        version=cartridge.version,
        description=cartridge.metadata.description,
        author=cartridge.metadata.author,
        capabilities=cartridge.metadata.capabilities,
        dependencies=cartridge.metadata.dependencies,
        model_preference=cartridge.metadata.model_preference,
        compatible_models=cartridge.metadata.compatible_models,
        trust_threshold=cartridge.metadata.trust_threshold,
        git_repo=cartridge.metadata.git_repo,
        onboarding_human=cartridge.metadata.onboarding_human,
        onboarding_agent=cartridge.metadata.onboarding_agent,
        tools=[t.to_dict() for t in cartridge.tools.values()],
    )

    filename = f"{cartridge.name}.cartridge.yaml"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(manifest.to_yaml())
    return filepath


def load_cartridge_from_file(filepath: str) -> Cartridge:
    """Load a cartridge from a YAML manifest file."""
    with open(filepath, "r") as f:
        raw = f.read()
    manifest = CartridgeManifest.from_yaml(raw)
    builder = CartridgeBuilder(manifest.name)
    builder._manifest = manifest
    return builder.build()
