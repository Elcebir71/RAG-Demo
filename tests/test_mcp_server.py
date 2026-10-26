"""Tests for the MCP wrapper around verify_math.

Skipped automatically when the 'mcp' package is not installed (for example in
CI), because the MCP server is a local-only tool for now.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mcp_server  # noqa: E402


def test_tool_is_registered_with_check_types():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = [tool.name for tool in tools]
    assert names == ["verify_math"]

    schema = tools[0].inputSchema
    assert "check_type" in schema["required"]
    schema_text = str(schema)
    for check_type in ("arithmetic", "equation_solution", "step_validation"):
        assert check_type in schema_text


def test_wrong_step_is_reported_through_the_tool():
    result = mcp_server.verify_math_tool(
        check_type="step_validation",
        steps=["3x - 5 = 7", "3x = 12", "x = 3"],
    )
    assert result["status"] == "wrong"
    assert result["failed_step"] == 3


def test_unsafe_input_stays_unverifiable_through_the_tool():
    result = mcp_server.verify_math_tool(check_type="arithmetic", input="__import__('os')")
    assert result["status"] == "unverifiable"
    assert result["parser_invoked"] is False
