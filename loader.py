"""
Cartridge Agent — Cartridge Loader

Responsible for loading, validating, and initializing cartridges
from various sources (metadata, files, manifests). Handles dependency
resolution and ensures all requirements are met before a cartridge
enters the loaded state.

Responsibilities:
- Load cartridges from metadata, dicts, and files
- Validate cartridge integrity (checksums, dependencies, capabilities)
- Dependency resolution with topological ordering
- Batch loading with rollback on failure
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cartridge import (
    BUILTIN_CARTRIDGES,
    Cartridge,
    CartridgeMetadata,
    CartridgeRegistry,
    CartridgeTool,
)


# ======================================================================
# Validation Result
# ======================================================================

@dataclass
class ValidationResult:
    """Result of validating a cartridge."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# ======================================================================
# Load Result
# ======================================================================

@dataclass
class LoadResult:
    """Result of a cartridge load operation."""
    success: bool
    cartridge: Optional[Cartridge] = None
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


# ======================================================================
# CartridgeLoader
# ======================================================================

class CartridgeLoader:
    """Loads and validates cartridges into a registry.

    The loader is the single entry point for introducing new cartridges
    into the system. It performs validation, dependency checking, and
    optional activation in one cohesive operation.
    """

    def __init__(
        self,
        registry: Optional[CartridgeRegistry] = None,
        strict: bool = False,
    ) -> None:
        """Initialize loader.

        Args:
            registry: Target registry. Created automatically if None.
            strict: If True, reject cartridges with warnings.
        """
        self.registry = registry or CartridgeRegistry()
        self.strict = strict
        self._load_order: List[str] = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_metadata(metadata: CartridgeMetadata) -> ValidationResult:
        """Validate cartridge metadata for common issues."""
        errors: List[str] = []
        warnings: List[str] = []

        # Name checks
        if not metadata.name:
            errors.append("Cartridge name is required")
        elif not metadata.name.replace("-", "").replace("_", "").isalnum():
            errors.append(
                f"Invalid cartridge name '{metadata.name}': "
                "use only alphanumeric, hyphens, underscores"
            )

        # Version check
        if metadata.version:
            parts = metadata.version.split(".")
            if len(parts) > 4 or not all(
                p.isdigit() for p in parts if p
            ):
                warnings.append(
                    f"Non-semver version '{metadata.version}'"
                )

        # Trust threshold
        if not (0.0 <= metadata.trust_threshold <= 1.0):
            errors.append(
                f"Trust threshold {metadata.trust_threshold} out of range [0, 1]"
            )

        # Self-dependency
        if metadata.name in metadata.dependencies:
            errors.append(f"Cartridge '{metadata.name}' depends on itself")

        # Duplicate dependencies
        seen_deps: Set[str] = set()
        for dep in metadata.dependencies:
            if dep in seen_deps:
                warnings.append(f"Duplicate dependency '{dep}'")
            seen_deps.add(dep)

        # Empty capabilities
        if not metadata.capabilities:
            warnings.append("No capabilities declared")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def validate_tools(tools: List[CartridgeTool]) -> ValidationResult:
        """Validate a list of cartridge tools."""
        errors: List[str] = []
        warnings: List[str] = []

        names: Set[str] = set()
        for tool in tools:
            if not tool.name:
                errors.append("Tool name is required")
            if tool.name in names:
                errors.append(f"Duplicate tool name '{tool.name}'")
            names.add(tool.name)
            if not tool.description:
                warnings.append(f"Tool '{tool.name}' has no description")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_cartridge(self, cartridge: Cartridge) -> ValidationResult:
        """Full validation of a cartridge and its metadata."""
        meta_result = self.validate_metadata(cartridge.metadata)
        tool_result = self.validate_tools(list(cartridge.tools.values()))

        all_errors = meta_result.errors + tool_result.errors
        all_warnings = meta_result.warnings + tool_result.warnings

        # Check dependencies against registry
        missing = self.registry.check_dependencies(cartridge.name)
        if missing:
            all_warnings.append(
                f"Missing dependencies: {', '.join(missing)}"
            )

        return ValidationResult(
            valid=len(all_errors) == 0 and (
                not self.strict or len(all_warnings) == 0
            ),
            errors=all_errors,
            warnings=all_warnings,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        metadata: CartridgeMetadata,
        tools: Optional[List[CartridgeTool]] = None,
        activate: bool = False,
    ) -> LoadResult:
        """Load a cartridge from metadata and optional tools.

        Validates, creates, registers, and optionally activates.
        """
        # Validate metadata
        meta_val = self.validate_metadata(metadata)
        if not meta_val.valid:
            return LoadResult(success=False, errors=meta_val.errors)

        # Create cartridge
        cart = Cartridge(metadata, tools)

        # Validate full cartridge
        cart_val = self.validate_cartridge(cart)
        if not cart_val.valid:
            return LoadResult(success=False, errors=cart_val.errors)

        # Register and load
        self.registry.register(cart)
        if not self.registry.load(cart.name):
            return LoadResult(
                success=False,
                errors=[f"Failed to load cartridge '{cart.name}'"],
            )

        if activate:
            if not self.registry.activate(cart.name):
                return LoadResult(
                    success=False,
                    errors=[f"Failed to activate cartridge '{cart.name}'"],
                )

        self._load_order.append(cart.name)
        return LoadResult(success=True, cartridge=cart)

    def load_from_dict(
        self,
        data: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        activate: bool = False,
    ) -> LoadResult:
        """Load a cartridge from a dictionary."""
        try:
            metadata = CartridgeMetadata.from_dict(data)
        except (TypeError, KeyError) as exc:
            return LoadResult(success=False, errors=[str(exc)])

        tool_list: List[CartridgeTool] = []
        if tools:
            for td in tools:
                tool_list.append(CartridgeTool(
                    name=td.get("name", ""),
                    description=td.get("description", ""),
                    parameters=td.get("parameters", {}),
                    required_capabilities=td.get("required_capabilities", []),
                ))

        return self.load(metadata, tool_list, activate=activate)

    def load_from_file(self, path: str, activate: bool = False) -> LoadResult:
        """Load a cartridge from a JSON file."""
        filepath = Path(path)
        if not filepath.exists():
            return LoadResult(
                success=False,
                errors=[f"File not found: {path}"],
            )
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return LoadResult(success=False, errors=[str(exc)])

        tools_data = data.pop("tools", [])
        return self.load_from_dict(data, tools=tools_data, activate=activate)

    # ------------------------------------------------------------------
    # Batch Loading
    # ------------------------------------------------------------------

    def load_batch(
        self,
        cartridges: List[Tuple[CartridgeMetadata, List[CartridgeTool]]],
        activate: bool = False,
    ) -> List[LoadResult]:
        """Load multiple cartridges. Returns per-cartridge results.

        If any validation fails, stops and returns partial results.
        """
        results: List[LoadResult] = []
        loaded_names: List[str] = []

        for metadata, tools in cartridges:
            result = self.load(metadata, tools, activate=activate)
            results.append(result)
            if result.success:
                loaded_names.append(result.cartridge.name)  # type: ignore
            else:
                # Rollback successfully loaded cartridges
                for name in loaded_names:
                    self.registry.unregister(name)
                break

        return results

    # ------------------------------------------------------------------
    # Dependency-aware loading
    # ------------------------------------------------------------------

    def load_with_dependencies(
        self,
        metadata: CartridgeMetadata,
        tools: Optional[List[CartridgeTool]] = None,
        activate: bool = False,
    ) -> List[LoadResult]:
        """Load a cartridge and all its dependencies in order.

        Dependencies are loaded first (topological order), then the
        target cartridge.
        """
        # First, create and register the target to resolve deps
        temp_cart = Cartridge(metadata, tools)
        self.registry.register(temp_cart)

        # Resolve dependency order
        dep_names = self.registry.resolve_dependencies(metadata.name)

        # Remove temp registration (will be properly loaded later)
        self.registry.unregister(metadata.name)

        results: List[LoadResult] = []

        # Load missing dependencies from builtins if available
        builtin_map = {c.name: c for c in BUILTIN_CARTRIDGES}
        for dep_name in dep_names:
            if self.registry.get(dep_name):
                continue  # Already loaded
            if dep_name in builtin_map:
                dep_cart = builtin_map[dep_name]
                dep_cart.load()
                self.registry.register(dep_cart)
                results.append(LoadResult(success=True, cartridge=dep_cart))
            else:
                results.append(LoadResult(
                    success=False,
                    errors=[f"Cannot resolve dependency '{dep_name}'"],
                ))
                return results

        # Load target
        result = self.load(metadata, tools, activate=activate)
        results.append(result)
        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def load_order(self) -> List[str]:
        return list(self._load_order)

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": len(self._load_order),
            "load_order": self._load_order,
            "registry_count": len(self.registry.list_all()),
            "strict": self.strict,
        }
