# cartridge-agent

Pluggable, hot-swappable agent cartridge modules. A cartridge is a self-contained behavior module that can be loaded, activated, swapped, and ejected at runtime without restarting the host system.

## Concepts

**Cartridge** — A loadable capability module with metadata, tools, and a managed lifecycle.

**Slot** — A named dock where a cartridge lives. Supports insert, eject, lock, and atomic swap.

**Loader** — Validates and loads cartridges with dependency resolution and batch support.

**HotSwapManager** — Zero-downtime cartridge replacement with prepare → commit → rollback protocol.

**Registry** — Central catalog of all cartridges with event hooks and dependency graphs.

**Bridge** — Connects cartridges to a MUD-style room architecture with skins and command routing.

**Scene** — Composes multiple cartridges into a unified interaction context.

## Quick Start

```python
from cartridge import Cartridge, CartridgeMetadata, CartridgeTool, CartridgeRegistry

# Create a cartridge
meta = CartridgeMetadata(
    name="my-agent",
    version="1.0.0",
    description="My custom agent cartridge",
    capabilities=["chat", "tools"],
)
tools = [
    CartridgeTool("greet", "Say hello", handler=lambda name: f"Hello, {name}!"),
    CartridgeTool("add", "Add numbers", handler=lambda a, b: a + b),
]
cart = Cartridge(meta, tools)

# Lifecycle
cart.load()       # UNLOADED → LOADED
cart.activate()   # LOADED → ACTIVE
print(cart.execute("greet", "World"))  # "Hello, World!"
print(cart.execute("add", 3, 4))       # 7
cart.deactivate()  # ACTIVE → LOADED
cart.unload()      # LOADED → UNLOADED
```

## Slots

```python
from slot import Slot, SlotManager

mgr = SlotManager()
slot = mgr.create_slot("primary", max_trust=0.8)

slot.insert(cart)
slot.lock("processing")   # Prevent ejection during critical work
slot.unlock()
ejected = slot.eject()    # Returns the cartridge
```

## Hot-Swap

```python
from swap import HotSwapManager

reg = CartridgeRegistry()
reg.register(cart)
reg.load("my-agent")
reg.activate("my-agent")

# Swap to a new version without downtime
swapper = HotSwapManager(reg)
new_meta = CartridgeMetadata(name="my-agent", version="2.0.0")
record = swapper.swap("my-agent", new_meta)
print(record.phase)      # SwapPhase.COMPLETED
print(record.duration_ms)  # Time taken in milliseconds
```

## Loader with Dependencies

```python
from loader import CartridgeLoader

loader = CartridgeLoader(registry=reg)
result = loader.load(
    CartridgeMetadata(name="worker", dependencies=["my-agent"]),
    tools=[CartridgeTool("process", "Process data")],
)
print(result.success)  # True

# Load with full dependency resolution
results = loader.load_with_dependencies(
    CartridgeMetadata(name="pipeline", dependencies=["worker", "my-agent"]),
)
```

## CLI

```bash
# List cartridges
cartridge-agent list

# Load a cartridge
cartridge-agent load navigation

# Build from template
cartridge-agent build --template relay --name my-relay

# Create a scene
cartridge-agent scene create --name ops --cartridges spreader-loop,oracle-relay

# Status
cartridge-agent status -v
```

## Module Overview

| Module | Description |
|---|---|
| `cartridge.py` | Core Cartridge, CartridgeMetadata, CartridgeRegistry |
| `slot.py` | Slot and SlotManager for cartridge docking |
| `loader.py` | CartridgeLoader with validation and dependency resolution |
| `swap.py` | HotSwapManager for zero-downtime replacement |
| `bridge.py` | CartridgeBridge with MUD room and skin support |
| `scene.py` | SceneManager for multi-cartridge composition |
| `cartridge_builder.py` | DSL builder, templates, packaging |
| `cli.py` | Command-line interface |

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
