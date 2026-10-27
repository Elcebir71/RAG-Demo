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
    "unit_conversion",
    "percentage_change",
    "inequality_solution",
    "system_solution",
    "function_notation",
    "statistics",
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
    Without an answer, arithmetic, formula_substitution, unit_conversion and
    percentage_change only calculate and return 'unverifiable' with reason 'no_claim_to_check'.

    check_type:
      - arithmetic: input is a numeric expression, e.g. "-3/4 + 2/3 × (-1,5)";
        submitted_answer is the claimed value, e.g. "-7/4".
      - answer_equivalence: input and submitted_answer are two notations of one
        value, e.g. "-7/4" and "-1,75".
      - expression_equivalence: input is an identity to check,
        e.g. "-2(x - 3) = -2x + 6" or "2(a + b) = 2a + 2b".
      - equation_solution: input is an equation, submitted_answer is the
        claimed solution, e.g. "x = 4", "geen oplossing" or "alle getallen".
        abc-formule answers: "x = (-3 ± √13)/2". misconception 'sign_of_b':
        b used instead of -b; 'divided_by_2_not_2a': divided by 2 instead of
        2a. The discriminant: formula_substitution with "D = b² - 4ac".
      - step_validation: steps is the list of equation lines, the original
        equation first, e.g. ["3x - 5 = 7", "3x = 12", "x = 4"].
      - formula_substitution: input is a letter formula, substitutions the
        values to fill in, submitted_answer the claimed result,
        e.g. "w = 4p + 10", {"p": "3"}, "w = 22". The unknown may also be in
        a denominator: "v = s / t", {"v": "60", "s": "120"}, "t = 2".
      - formula_rewrite: input is a formula, submitted_answer the formula
        rewritten for another letter, e.g. "w = 4p + 10" and "p = (w - 10) / 4".
        With letters in a denominator the result lists conditions such as
        "t ≠ 0"; tell the teacher these conditions.
      - unit_conversion: input and submitted_answer are quantities with units,
        e.g. "72 km/u" and "20 m/s", or "1 dm³" and "1 L". Supported: metric
        length, mass, area, volume (litres included, ml or mL), time (s, min,
        u or h, d) and speed (m/s, km/u, km/h, m/min). Pass a rounding
        instruction in task_instruction when the task asks for rounding.
      - percentage_change: input is "old -> new", submitted_answer the
        student's percentage, e.g. "80 -> 100" and "25%"; a decrease as "-20%"
        or "20% afname". A decrease written as plain "20%" returns 'wrong'
        with direction_missing; misconception 'divided_by_new_value' means the
        student divided by the new instead of the old value.
      - inequality_solution: input is a linear inequality in one letter,
        also a double one, e.g. "5 - 2x > 1" or "-4 < 2x + 2 ≤ 8" (<= and >=
        work too); submitted_answer the student's solution, e.g. "x < 2",
        "3 > x", "-3 < x ≤ 3", "geen oplossing" or "alle reële getallen".
        solution_set gives the correct set as text. misconception
        'direction_reversed': right bound, sign the wrong way (often a
        negative coefficient without flipping); 'wrong_bound_type': < and ≤
        mixed up. Interval notation (⟨←, 3⟩) returns
        interval_notation_not_supported: ask for the x < 3 form.
      - system_solution: input is two linear equations separated by ';' or
        a new line, e.g. "2x + y = 7; x - y = 2"; submitted_answer names both
        letters, e.g. "x = 3, y = 1", "x = 3 en y = 1", "geen oplossing" or
        "oneindig veel oplossingen". A bare pair "(3, 1)" is not accepted:
        rewrite it with the letters only if the student's order is certain.
        misconception 'values_swapped': right numbers, wrong letters;
        'missing_variable': one letter missing; satisfied_equations lists
        the equations (1, 2) the answer does fit.
      - function_notation: input is the definition(s) and then the task,
        separated by ';', e.g. "f(x) = 2x² - 3x + 1; f(-2)" (answer "15" or
        "f(-2) = 15"), "f(x) = 3x - 4; f(x) = 8" (answer "x = 4") or
        "f(x) = 2x + 1; g(x) = -x + 7; f(x) = g(x)". misconception
        'negative_base_not_in_brackets': the student computed -2² instead of
        (-2)²; argument_mismatch: the answer names another argument, e.g.
        f(2) instead of f(-2). Use only this check type for f(x) notation:
        elsewhere a(b + c) means a times (b + c).
      - statistics: input is the measure, a colon and the data, e.g.
        "gemiddelde: 4, 7, 7, 9, 3", "mediaan: 1,5; 2,5; 3,5". Measures:
        gemiddelde, mediaan, modus, spreidingsbreedte. Separate values with
        ';' or with a comma plus a space (a comma between digits is a
        decimal comma). Modus answers: "7", "5 en 8" or "geen modus".
        Misconceptions: median_not_sorted, median_even_count,
        confused_with_median, confused_with_mean.

    Percentages elsewhere: 25% = 25/100, "25% van 80" = 20. "80 + 15%" is
    ambiguous (calculator 92, algebra 80,15) and returns ambiguous_percentage:
    ask which one is meant. Rounding instructions ("Rond af op 2 decimalen",
    "Rond af op 1 decimaal") work for arithmetic, answer_equivalence,
    equation_solution, formula_substitution, unit_conversion and
    percentage_change.

    Letters: any single letter can be a variable (case-sensitive); 'ab' means
    a times b. e and i are variables too, but any power of e (e², e^x) and
    i² return ambiguous_constant: ask the teacher whether e is Euler's number
    or i the imaginary unit. π is always the constant pi. Word formulas
    ('aantal') are not supported. If an equation has more than one letter,
    pass the letter to solve for in 'variable' (also for inequalities). If the result has
    variable_mismatch with value_correct true, tell the teacher the value is
    right but the answer is written under the wrong letter (status is still
    'wrong').

    Roots: √ applies to the next number, letter or bracket (√12, 2√3,
    √(x + 1)). For root answers the result has is_simplified (2√3 yes, √12
    no, 1/√2 no): report it separately from the status. √ of a negative
    number returns not_real. Equations with two solutions: pass the student's
    answer as written, e.g. "x = √5 of x = -√5" or "x = ±√5". Pythagoras: use
    formula_substitution with "a² + b² = c²"; the positive length is taken
    (assumed_positive), and misconception forgot_square_root means the student
    gave c² instead of c.

    Notation: decimal comma (1,5), mixed numbers (1 3/4), ':' for division,
    '×' or '·' for multiplication. ':' and '/' are read strictly from left
    to right, also when a fraction follows: "2/3 : 4/5" is ((2/3) : 4) / 5.
    When a student divides by a fraction, put that fraction in brackets:
    "2/3 : (4/5)", "(-1/3)² : (2/9)". A euro sign in an answer (€288) is
    ignored for arithmetic. A period inside a number (1.000) is
    rejected as ambiguous.

    task_instruction: optional, e.g. "Rond af op 2 decimalen." Pass it only
    when the task itself asks for rounding.

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
