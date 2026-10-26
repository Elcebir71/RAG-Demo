"""Local MCP server that exposes verify_math to Claude Desktop.

Runs over stdio: Claude Desktop starts this script and talks to it through
stdin/stdout. Nothing is exposed to the network. Do not print to stdout in
this file; stdout is reserved for the MCP protocol.

Manual start (for a quick smoke test):
    py mcp_server.py
"""
from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from verify_math import verify_math

server = FastMCP("wiskunde-verify")

CheckType = Literal[
    "arithmetic",
    "answer_equivalence",
    "expression_equivalence",
    "equation_solution",
    "step_validation",
]


@server.tool(name="verify_math")
def verify_math_tool(
    check_type: CheckType,
    input: str = "",
    submitted_answer: str | None = None,
    steps: list[str] | None = None,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Independently check a maths claim with SymPy (Dutch school notation).

    Use this tool before presenting any calculated answer, answer key or worked
    solution to a teacher. Report the tool's status exactly as returned.

    check_type:
      - arithmetic: input is a numeric expression, e.g. "-3/4 + 2/3 × (-1,5)";
        submitted_answer is the claimed value, e.g. "-7/4".
      - answer_equivalence: input and submitted_answer are two notations of one
        value, e.g. "-7/4" and "-1,75".
      - expression_equivalence: input is an identity in x to check,
        e.g. "-2(x - 3) = -2x + 6".
      - equation_solution: input is an equation in x, submitted_answer is the
        claimed solution, e.g. "x = 4", "geen oplossing" or "alle getallen".
      - step_validation: steps is the list of equation lines, the original
        equation first, e.g. ["3x - 5 = 7", "3x = 12", "x = 4"].

    Notation: decimal comma (1,5), mixed numbers (1 3/4), ':' for division,
    '×' or '·' for multiplication, only the variable x. A period inside a
    number (1.000) is rejected as ambiguous.

    task_instruction: optional, e.g. "Rond af op 2 decimalen."

    Returns status 'verified', 'wrong' or 'unverifiable'. 'unverifiable' means
    the tool could not check the claim; never present such a claim as checked.
    'verified' only covers the mathematical property tested, not whether a
    word problem was modelled correctly.
    """
    return verify_math(
        check_type=check_type,
        input=input,
        submitted_answer=submitted_answer,
        steps=steps,
        task_instruction=task_instruction,
    )


if __name__ == "__main__":
    server.run()  # stdio transport
