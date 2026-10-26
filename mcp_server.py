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
    "formula_substitution",
    "formula_rewrite",
]


@server.tool(name="verify_math")
def verify_math_tool(
    check_type: CheckType,
    input: str = "",
    submitted_answer: str | None = None,
    steps: list[str] | None = None,
    task_instruction: str | None = None,
    variable: str | None = None,
    substitutions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Independently check a maths claim with SymPy (Dutch school notation).

    Use this tool before presenting any calculated answer, answer key or worked
    solution to a teacher. Report the tool's status exactly as returned.

    When checking a student's work, always pass the student's answer in
    submitted_answer exactly as the student wrote it; never correct it first.
    Without an answer, arithmetic and formula_substitution only calculate and
    return 'unverifiable' with reason 'no_claim_to_check'.

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
      - formula_substitution: input is a letter formula, substitutions the
        values to fill in, submitted_answer the claimed result,
        e.g. "w = 4p + 10", {"p": "3"}, "w = 22".
      - formula_rewrite: input is a formula, submitted_answer the formula
        rewritten for another letter, e.g. "w = 4p + 10" and "p = (w - 10) / 4".

    Letters: any single letter can be a variable (case-sensitive); 'ab' means
    a times b. e and i are variables too, but any power of e (e², e^x) and
    i² return ambiguous_constant: ask the teacher whether e is Euler's number or i the
    imaginary unit. π is always the constant pi. Word formulas ('aantal') are
    not supported. If an equation has more than one letter, pass the letter to
    solve for in 'variable'. If the result has variable_mismatch with
    value_correct true, tell the teacher the value is right but the answer is
    written under the wrong letter (status is still 'wrong').

    Notation: decimal comma (1,5), mixed numbers (1 3/4), ':' for division,
    '×' or '·' for multiplication. A period inside a
    number (1.000) is rejected as ambiguous.

    task_instruction: optional, e.g. "Rond af op 2 decimalen."

    For step_validation, tell the teacher every invalid step: failed_step is
    the first one, additional_invalid_steps lists later independent errors,
    steps_consistent_after_error lists steps that correctly carry an earlier
    error forward, and final_answer_correct says whether the final x = ...
    matches the original equation (it can be true even when steps are wrong).

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
        variable=variable,
        substitutions=substitutions,
    )


if __name__ == "__main__":
    server.run()  # stdio transport
