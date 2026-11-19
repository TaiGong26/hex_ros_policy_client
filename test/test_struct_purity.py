################################################################
# Copyright 2026 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2026-11-18
################################################################
"""Purity checks for the data layer.

``struct`` must stay importable without ROS, without the rest of the package
and without any client library, so that it can be tested and reused alone.
Only the standard library, ``numpy`` and modules of ``struct`` itself are
allowed.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

STRUCT_ROOT = (pathlib.Path(__file__).resolve().parents[1]
               / "hex_ros_policy_client" / "struct")
ALLOWED_THIRD_PARTY = {"numpy"}


def struct_modules() -> list[pathlib.Path]:
    """List the modules of the data layer.

    Returns:
        list: Paths of every ``*.py`` file under ``struct``.
    """
    return sorted(STRUCT_ROOT.glob("*.py"))


def imported_roots(path: pathlib.Path) -> set[str]:
    """Collect the roots of the absolute imports of one module.

    Args:
        path (pathlib.Path): Module file to inspect.

    Returns:
        set: Dotted root names, for example ``{"numpy", "dataclasses"}``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def relative_targets(path: pathlib.Path) -> set[str]:
    """Collect the sibling modules reached by relative imports.

    Args:
        path (pathlib.Path): Module file to inspect.

    Returns:
        set: Module names imported from the same layer.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                targets.add(node.module.split(".")[0])
    return targets


@pytest.mark.parametrize("path", struct_modules(), ids=lambda p: p.name)
def test_only_stdlib_and_numpy(path: pathlib.Path) -> None:
    """A data module imports nothing but the standard library and numpy.

    Args:
        path (pathlib.Path): Module file under test.
    """
    allowed = set(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY
    forbidden = imported_roots(path) - allowed
    assert forbidden == set()


@pytest.mark.parametrize("path", struct_modules(), ids=lambda p: p.name)
def test_relative_imports_stay_inside_struct(path: pathlib.Path) -> None:
    """A data module may only reach its own siblings.

    Args:
        path (pathlib.Path): Module file under test.
    """
    siblings = {item.stem for item in struct_modules()}
    assert relative_targets(path) <= siblings


def test_no_ros_import() -> None:
    """No data module pulls in ROS or the message package."""
    for path in struct_modules():
        roots = imported_roots(path)
        assert "rclpy" not in roots
        assert "hex_ros_msgs" not in roots
        assert "hex_ros_policy_client" not in roots
