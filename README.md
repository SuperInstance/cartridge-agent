# cartridge-agent — Standalone Cartridge Agent

**A standalone agent powered by swappable behavior cartridges. Load capabilities, not code.**

## What This Gives You

- **Cartridge loading** — dynamically load behavior modules (debugging, profiling, docs, testing)
- **Slot management** — multiple cartridge slots with priority ordering
- **Scene-based behavior** — switch cartridge sets based on the current "scene" (development, production, debugging)
- **Bridge interface** — communicate with fleet services and other cartridge agents
- **CLI** — `cartridge-agent load <name>`, `cartridge-agent swap <old> <new>`

## Quick Start

```bash
pip install cartridge-agent
```

```python
from cartridge_agent import CartridgeAgent

agent = CartridgeAgent(id="worker-3")

# Load cartridges
agent.load("rust-builder")
agent.load("docs-writer")
agent.load("benchmark-runner")

# Run a task — auto-selects the right cartridge
result = agent.execute("Build the Rust library")
print(result.cartridge_used)  # "rust-builder"

# Swap cartridges at runtime
agent.swap("docs-writer", "api-docs-writer")

# Scene-based switching
agent.set_scene("debugging")
# Automatically loads: debugging-cartridge, logging-cartridge
```

## API Reference

### `CartridgeAgent(id)` — `load(cartridge)`, `swap(old, new)`, `execute(task)`, `set_scene(scene)`
### `Slot(priority, cartridge)` — Ordered cartridge slots
### `Scene(name, cartridges)` — Named set of cartridges
### `Bridge` — Fleet communication interface

## How It Fits

A standalone agent that uses [cartridge-mcp](https://github.com/SuperInstance/cartridge-mcp) for cartridge management in the [SuperInstance fleet](https://github.com/SuperInstance).

- **[cartridge-mcp](https://github.com/SuperInstance/cartridge-mcp)** — MCP server for cartridge management
- **[agent-forge](https://github.com/SuperInstance/agent-forge)** — Universal agent framework
- **[claude-code-vessel](https://github.com/SuperInstance/claude-code-vessel)** — Containerized execution

## Testing

```bash
pytest tests/
```

## Installation

```bash
pip install cartridge-agent
```

Python 3.10+. MIT license.
