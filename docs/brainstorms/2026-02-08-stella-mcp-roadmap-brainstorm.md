# Stella-MCP Roadmap Brainstorm

**Date:** 2026-02-08

## Problem Statement

Stella-MCP can create and validate system dynamics models, but the generated layouts are cluttered — overlapping elements, tight spacing, flow arrows crossing through stocks. Models need manual rearrangement in Stella before they're usable. Additionally, there's no way to start from common model patterns or organize large models into modules.

## What We're Building (Priority Order)

### 1. Force-Directed Layout Engine (Highest Priority)

Replace the current phase-based layout (place → detect collisions → nudge) with a force-directed (spring model) algorithm.

**Why force-directed over hierarchical:**
- System dynamics models often have feedback loops and circular topologies
- Force-directed naturally produces grid-like and circular arrangements
- Adapts to the topology rather than imposing a structure
- Better default for mixed chain + loop models (common in biogeochemistry)

**Domain-specific tweaks to add:**
- Keep stocks at similar Y-levels when they share flows
- Give flow arrows directional preference
- Ensure connector arrows route cleanly around elements
- Minimum spacing guarantees between all element types

**Success criteria:**
- Models with 5-15 elements should open in Stella with no overlaps
- Feedback loops should render as circles or grids, not tangles
- Flow arrows should not cross through stocks or auxiliaries
- Sufficient breathing room between all elements

### 2. Model Templates (Medium Priority)

**Built-in templates** shipped with the MCP:
- Exponential growth
- SIR (epidemiology)
- Lotka-Volterra (predator-prey)
- Carbon cycle (2-3 box)
- Nutrient box model (2-box ocean)

**User-defined templates:**
- `save_as_template` — Save current model as a reusable template
- `load_template` — Load a built-in or user-defined template as the working model
- `list_templates` — Show available templates (built-in + user-defined)

Templates are just `.stmx` files stored in a known directory.

### 3. Submodels/Modules (Future)

- `create_module` / `add_to_module` tools
- Map to XMILE `<group>` or `<model>` elements
- Essential for larger real-world models (e.g., coupled ocean-atmosphere)
- Needs careful XMILE spec research before implementation

## Key Decisions

- **Layout algorithm:** Force-directed (spring model), not hierarchical
- **Template storage:** Built-in + user-defined, both as `.stmx` files
- **Priority order:** Layout quality → Templates → Submodels
- **Model editing (delete/update/rename):** Deferred for now

## Open Questions

- What force-directed library to use (pure Python implementation vs. dependency like NetworkX)?
- Where to store user-defined templates (project-local vs. global `~/.stella-mcp/templates/`)?
- How many layout iterations / what convergence threshold for force simulation?
- Should templates include pre-set time specs or let the user configure those on load?
