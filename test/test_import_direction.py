################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Static checks for the package import direction.

The package is split into seven layers with a single allowed direction, from
the node down to the data layer: ``policy_client``, ``core``, ``process``,
``protocol``, ``device``, ``utils`` and ``struct``. A module may import its own
layer, any layer below it, and nothing else. Modules inside ``struct`` are a
pure data layer: they may only import their own siblings, never another
layer.

``protocol`` and ``device`` are two adapters of the same height, one towards
the model side and one towards ROS. A linear order cannot express that they
are peers, so they are also listed in a peer group: neither may import the
other, which keeps a change of carrier from reaching the device layer and the
reverse.

Attributes:
    PACKAGE: Importable name of the package.
    SOURCE_ROOT: Directory that holds the six sub layers.
    LAYER_ORDER: Layers from the lowest to the highest.
    PEER_GROUPS: Layer groups whose members may not import each other.
    NODE_LAYER: Layer of the modules that live directly in the package root.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = "hex_ros_policy_client"
SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1] / PACKAGE
LAYER_ORDER = ("struct", "utils", "device", "protocol", "process", "core",
               "policy_client")
PEER_GROUPS = (("device", "protocol"),)
NODE_LAYER = "policy_client"


def iter_modules() -> list[tuple[pathlib.Path, str]]:
    """Collect every module of the package with the layer it belongs to.

    Returns:
        list: Pairs of (module path, layer name). Modules stored directly in
        the package root belong to the node layer.
    """
    modules = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        parts = path.relative_to(SOURCE_ROOT).parts
        modules.append((path, parts[0] if len(parts) > 1 else NODE_LAYER))
    return modules


def resolve_targets(path: pathlib.Path,
                    tree: ast.AST) -> list[tuple[str, ...]]:
    """Resolve every import of this package to package relative tokens.

    Args:
        path (pathlib.Path): Module file being inspected.
        tree (ast.AST): Parsed syntax tree of that file.

    Returns:
        list: One tuple of dotted tokens per imported target, relative to the
        package root. Imports of other packages are skipped.
    """
    package_parts = path.relative_to(SOURCE_ROOT).parts[:-1]
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(
                tuple(alias.name.split(".")[1:])
                for alias in node.names
                if alias.name == PACKAGE
                or alias.name.startswith(PACKAGE + ".")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - node.level + 1
                base = package_parts[:max(keep, 0)]
            elif node.module and node.module.split(".")[0] == PACKAGE:
                base = ()
            else:
                continue
            module_parts = tuple(node.module.split(".")) if node.module else ()
            if node.level:
                targets.append(base + module_parts)
            else:
                targets.append(module_parts[1:])
    return targets


def target_layer(tokens: tuple[str, ...]) -> str:
    """Map resolved tokens to the layer of the imported module.

    Args:
        tokens (tuple): Package relative tokens of the imported module.

    Returns:
        str: Layer name; package root modules belong to the node layer.
    """
    if tokens and tokens[0] in LAYER_ORDER:
        return tokens[0]
    return NODE_LAYER


def layer_rank(layer: str) -> int:
    """Return the height of a layer, used for the direction comparison.

    Args:
        layer (str): Layer name.

    Returns:
        int: Rank that grows with the layer height.
    """
    return LAYER_ORDER.index(layer)


def peers_of(layer: str) -> frozenset:
    """Return the layers that sit beside this one and may not be imported.

    Args:
        layer (str): Layer name.

    Returns:
        frozenset: Peer layers, empty when the layer has no peer.
    """
    for group in PEER_GROUPS:
        if layer in group:
            return frozenset(group)
    return frozenset()


def find_violations(path: pathlib.Path, layer: str) -> list[str]:
    """Report imports that break the layer direction for one module.

    Args:
        path (pathlib.Path): Module file being inspected.
        layer (str): Layer the module belongs to.

    Returns:
        list: Human readable violation messages, empty when the module is
        compliant.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for tokens in resolve_targets(path, tree):
        imported = target_layer(tokens)
        name = ".".join(tokens) or "<package>"
        if layer == "struct":
            if imported != "struct":
                violations.append(
                    f"struct module {path.name} imports layer {name}")
        elif layer_rank(imported) > layer_rank(layer):
            violations.append(
                f"{layer} module {path.name} imports higher layer {name}")
        elif imported != layer and imported in peers_of(layer):
            violations.append(
                f"{layer} module {path.name} imports sibling adapter {name}")
    return violations


@pytest.mark.parametrize("path, layer", iter_modules(),
                         ids=lambda item: str(item))
def test_import_direction(path: pathlib.Path, layer: str) -> None:
    """Every module only imports itself or a layer below it.

    Args:
        path (pathlib.Path): Module file under test.
        layer (str): Layer the module belongs to.
    """
    assert find_violations(path, layer) == []


def test_all_layers_present() -> None:
    """Every layer is represented by at least one module."""
    seen = {layer for _, layer in iter_modules()}
    assert seen == set(LAYER_ORDER)


def test_peer_adapters_are_detected() -> None:
    """The peer rule reports both directions between the two adapters."""
    assert peers_of("device") == frozenset({"device", "protocol"})
    assert peers_of("protocol") == frozenset({"device", "protocol"})
    assert peers_of("core") == frozenset()
