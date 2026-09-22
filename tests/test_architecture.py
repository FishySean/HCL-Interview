"""The layering rule, enforced instead of documented.

Decoupling that lives only in a README decays on the first deadline. These
tests parse the import statements of every module and fail if a layer reaches
sideways, so the architecture diagram cannot silently stop being true.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "robot_runtime"

# Layers communicate through the bus and are wired together in the composition
# root. None of them may know that any of the others exists.
LAYERS = ("perception", "presence_filter", "behavior", "safety", "motion")

# Shared vocabulary and infrastructure, which every layer is allowed to use.
SHARED = ("contracts", "runtime")

# The composition root is the one place allowed to import everything: that is
# what a composition root is for.
EXEMPT = {PACKAGE / "runtime" / "app.py", PACKAGE / "cli.py"}


def modules_of(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).rglob("*.py"))


def imported_packages(path: Path) -> set[str]:
    """Top-level `robot_runtime` subpackages referenced by this module.

    Relative imports are resolved against the module's own position, which is
    what makes `from ..safety.gate import ...` visible as a dependency on
    `safety` rather than as an opaque string.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parts = path.relative_to(PACKAGE).parts[:-1]
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("robot_runtime."):
                    found.add(node.module.split(".")[1])
                continue
            # level 1 == current package, level 2 == parent, and so on.
            base = parts[: len(parts) - (node.level - 1)]
            target = (*base, *(node.module.split(".") if node.module else ()))
            if target:
                found.add(target[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("robot_runtime."):
                    found.add(alias.name.split(".")[1])
    return found


@pytest.mark.parametrize("layer", LAYERS)
def test_a_layer_never_imports_another_layer(layer: str):
    forbidden = set(LAYERS) - {layer}
    offences: list[str] = []

    for module in modules_of(layer):
        if module in EXEMPT:
            continue
        reached = imported_packages(module) & forbidden
        if reached:
            offences.append(f"{module.relative_to(PACKAGE)} imports {sorted(reached)}")

    assert not offences, (
        "layers must talk through contracts and the bus, never directly:\n  "
        + "\n  ".join(offences)
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_a_layer_only_reaches_for_shared_packages(layer: str):
    allowed = {layer, *SHARED}
    for module in modules_of(layer):
        if module in EXEMPT:
            continue
        stray = imported_packages(module) - allowed
        assert not stray, f"{module.relative_to(PACKAGE)} imports unexpected {sorted(stray)}"


def test_contracts_depends_on_nothing_inside_the_project():
    """If the shared vocabulary could import a layer, every layer would depend
    on that layer transitively and the whole scheme would be decorative."""
    for module in modules_of("contracts"):
        assert imported_packages(module) <= {"contracts"}, module.name


def test_the_composition_root_is_the_only_place_that_knows_every_layer():
    wiring = imported_packages(PACKAGE / "runtime" / "app.py")
    assert set(LAYERS) <= wiring


def test_the_motion_layer_cannot_subscribe_to_unapproved_commands():
    """The safety gate is non-bypassable because of who subscribes to what:
    motion listens for ApprovedMotion, and only the gate publishes those."""
    sources = "\n".join(path.read_text(encoding="utf-8") for path in modules_of("motion"))
    assert "ApprovedMotion" in sources
    assert "subscribe(MotionCommand" not in sources

    gate_sources = "\n".join(path.read_text(encoding="utf-8") for path in modules_of("safety"))
    assert "subscribe(MotionCommand" in gate_sources
