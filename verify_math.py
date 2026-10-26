"""A small, conservative mathematics verifier for Dutch classroom notation.

The module intentionally does NOT use sympy.parse_expr or Python eval. A tiny
recursive-descent parser accepts a restricted mathematical grammar and builds
SymPy objects directly. Unsupported or ambiguous input fails closed as
'unverifiable'.

Scope in this first version:
- exact numeric arithmetic and answer-equivalence checks;
- symbolic identity checks in one variable x;
- real solution-set checks for equations in x;
- step-by-step equation equivalence checks.

This is a verification aid, not a proof of pedagogical correctness or of the
semantic modelling of a word problem. See README.md for the limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re
from typing import Any, Iterable, Sequence

import sympy as sp
from sympy.core.sorting import default_sort_key


__all__ = ["verify_math", "MAX_INPUT_LENGTH", "MAX_TOKENS"]

MAX_INPUT_LENGTH = 256
MAX_TOKENS = 128
MAX_NUMERIC_DIGITS = 32
MAX_POWER_ABS = 20
X = sp.Symbol("x", real=True)

# Keep the allowlist intentionally small. Letters other than x are not part of
# the expression grammar; Dutch prose is classified separately as a semantic
# formalisation task. Quotes, underscores and other executable-looking syntax
# are rejected before the parser is invoked.
_ALLOWED_CHARS = re.compile(r"^[0-9xX+\-−–—*/:×·()^²³=,\.\s]*$")
_SUSPICIOUS_CHARS = set("_'\"[]{}!@#$%&`\\;")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_MIXED_RE = re.compile(r"\d+\s+\d+\s*/\s*\d+")
_NUMBER_RE = re.compile(r"(?:\d+[,.]\d+|\d+)")
_AMBIGUOUS_DOT_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)+(?![\w.])")


class _MathInputError(ValueError):
    def __init__(self, reason: str, message_nl: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.message_nl = message_nl or _REASON_NL.get(reason, "Deze invoer kan niet betrouwbaar worden gecontroleerd.")


_REASON_NL = {
    "input_too_long": "De invoer is langer dan de toegestane limiet.",
    "disallowed_characters_or_syntax": "De invoer bevat tekens of syntaxis die niet zijn toegestaan.",
    "requires_semantic_formalisation": "Zet de tekst eerst om in een formule of vergelijking; dat kan de tool niet zelfstandig valideren.",
    "ambiguous_comma_notation": "Deze komma-notatie is dubbelzinnig: het kan een decimaal getal of een coördinaat zijn.",
    "ambiguous_dot_notation": "Puntnotatie in een getal is dubbelzinnig in Nederlandse notatie; gebruik een komma voor decimalen.",
    "variable_denominator_unsupported": "Een noemer die van x afhangt kan domeinbeperkingen geven; deze invoer wordt niet automatisch vereenvoudigd.",
    "unsupported_variable": "Deze verifier ondersteunt alleen de variabele x; andere variabelen worden niet gecontroleerd.",
    "division_by_zero": "De uitdrukking bevat een deling door nul en heeft daarom geen geldige waarde.",
    "unsupported_math_form": "Deze wiskundige notatie valt buiten de ondersteunde grammatica.",
    "not_numeric_expression": "De uitdrukking bevat een variabele en is geen zuiver numerieke berekening.",
    "unsupported_solution_set": "De oplossing kan niet betrouwbaar worden bepaald met deze verifier.",
    "invalid_equation": "Voer precies één vergelijking met een gelijkteken in.",
    "invalid_answer": "Het antwoord kon niet als een ondersteunde wiskundige waarde worden gelezen.",
    "rounding_instruction_not_supported": "De afrondingsinstructie wordt niet ondersteund.",
}

_PREFLIGHT_REASONS = {
    "input_too_long",
    "disallowed_characters_or_syntax",
    "ambiguous_comma_notation",
    "ambiguous_dot_notation",
    "requires_semantic_formalisation",
    "unsupported_variable",
}


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    start: int
    end: int


def _preflight(text: Any) -> tuple[bool, str | None]:
    """Reject dangerous, overlong, prose, or ambiguous input before parsing."""
    if not isinstance(text, str) or not text.strip():
        return False, "disallowed_characters_or_syntax"
    if len(text) > MAX_INPUT_LENGTH:
        return False, "input_too_long"
    if any(char in _SUSPICIOUS_CHARS for char in text):
        return False, "disallowed_characters_or_syntax"

    # Dutch notation uses a comma for decimals. A period inside a numeric
    # token may be a thousands separator (1.000) or foreign decimal notation;
    # never reinterpret it silently as the integer before the period.
    if _AMBIGUOUS_DOT_NUMBER_RE.search(text):
        return False, "ambiguous_dot_notation"

    # (1,5) by itself is ambiguous in this context: coordinate or decimal.
    # A negative decimal in an arithmetic expression, e.g. (-1,5), remains
    # supported because the entire input is not just a coordinate-like pair.
    if re.fullmatch(r"\(\s*\d+\s*,\s*\d+\s*\)", text.strip()):
        return False, "ambiguous_comma_notation"

    if not _ALLOWED_CHARS.fullmatch(text):
        words = _WORD_RE.findall(text)
        # When the input clearly looks like a compact equation but contains a
        # variable other than x, report that actionable limitation instead of
        # misclassifying it as a word problem.
        if (words and any(word.casefold() != "x" for word in words)
                and "=" in text and not re.search(r"[.!?]", text) and len(words) <= 3):
            return False, "unsupported_variable"
        if words and all(word.casefold() == "x" for word in words):
            return False, "disallowed_characters_or_syntax"
        # Natural language should be formalised by a human/model upstream;
        # this function must not invent an equation from it.
        if words and not any(char in _SUSPICIOUS_CHARS for char in text):
            return False, "requires_semantic_formalisation"
        return False, "disallowed_characters_or_syntax"

    # 'x' is the only supported variable. The allowed-character regex already
    # excludes all other letters, so no separate names/functions can enter.
    return True, None


def _tokenize(text: str) -> list[_Token]:
    """Tokenise a restricted expression without eval/parse_expr."""
    # Unicode operator normalisation is explicit and local.
    text = text.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "²": "^2", "³": "^3"}))
    tokens: list[_Token] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
            continue

        mixed = _MIXED_RE.match(text, pos)
        if mixed:
            raw = mixed.group(0)
            if sum(c.isdigit() for c in raw) > MAX_NUMERIC_DIGITS:
                raise _MathInputError("unsupported_math_form")
            tokens.append(_Token("MIXED", raw, pos, mixed.end()))
            pos = mixed.end()
            continue

        number = _NUMBER_RE.match(text, pos)
        if number:
            raw = number.group(0)
            if sum(c.isdigit() for c in raw) > MAX_NUMERIC_DIGITS:
                raise _MathInputError("unsupported_math_form")
            tokens.append(_Token("NUMBER", raw, pos, number.end()))
            pos = number.end()
            continue

        if char in "xX":
            tokens.append(_Token("VAR", "x", pos, pos + 1))
            pos += 1
        elif char == "(":
            tokens.append(_Token("LPAREN", char, pos, pos + 1))
            pos += 1
        elif char == ")":
            tokens.append(_Token("RPAREN", char, pos, pos + 1))
            pos += 1
        elif char in "+-*/:^=":
            tokens.append(_Token("OP", "/" if char == ":" else char, pos, pos + 1))
            pos += 1
        else:
            # This should ordinarily be blocked by preflight too.
            raise _MathInputError("disallowed_characters_or_syntax")

        if len(tokens) > MAX_TOKENS:
            raise _MathInputError("unsupported_math_form")

    if not tokens:
        raise _MathInputError("unsupported_math_form")
    return tokens


class _Parser:
    """Recursive-descent parser for arithmetic and expressions in x."""
    def __init__(self, text: str):
        self.tokens = _tokenize(text)
        self.index = 0

    def _peek(self) -> _Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> _Token:
        token = self._peek()
        if token is None:
            raise _MathInputError("unsupported_math_form")
        self.index += 1
        return token

    def parse(self) -> sp.Expr:
        result = self._parse_expression()
        if self._peek() is not None:
            token = self._peek()
            if token and token.kind == "OP" and token.value == "=":
                raise _MathInputError("invalid_equation")
            raise _MathInputError("unsupported_math_form")
        return result

    def _parse_expression(self) -> sp.Expr:
        result = self._parse_term()
        while (tok := self._peek()) is not None and tok.kind == "OP" and tok.value in ("+", "-"):
            op = self._take().value
            rhs = self._parse_term()
            result = sp.Add(result, rhs if op == "+" else -rhs, evaluate=True)
        return result

    def _can_implicit_multiply(self) -> bool:
        token = self._peek()
        # Supports 2x, 2(...), x(...), and (...)(...). It intentionally does
        # not treat adjacency of two numeric tokens as multiplication: that
        # space is reserved for mixed-number notation (e.g. 1 3/4).
        return token is not None and token.kind in ("VAR", "LPAREN")

    def _parse_term(self) -> sp.Expr:
        result = self._parse_unary()
        while True:
            tok = self._peek()
            if tok is not None and tok.kind == "OP" and tok.value in ("*", "/"):
                op = self._take().value
                rhs = self._parse_unary()
                if op == "/":
                    if rhs == 0:
                        raise _MathInputError("division_by_zero")
                    # SymPy may simplify x/x to 1 and erase x != 0. To avoid
                    # accidentally asserting an identity over an invalid domain,
                    # fail closed whenever a denominator still depends on x.
                    if rhs.has(X):
                        raise _MathInputError("variable_denominator_unsupported")
                    result = sp.Mul(result, sp.Pow(rhs, -1, evaluate=True), evaluate=True)
                else:
                    result = sp.Mul(result, rhs, evaluate=True)
            elif self._can_implicit_multiply():
                rhs = self._parse_unary()
                result = sp.Mul(result, rhs, evaluate=True)
            else:
                break
        return result

    def _parse_unary(self) -> sp.Expr:
        tok = self._peek()
        if tok is not None and tok.kind == "OP" and tok.value in ("+", "-"):
            op = self._take().value
            value = self._parse_unary()
            return value if op == "+" else -value
        return self._parse_power()

    def _parse_power(self) -> sp.Expr:
        base = self._parse_atom()
        tok = self._peek()
        if tok is not None and tok.kind == "OP" and tok.value == "^":
            self._take()
            exponent = self._parse_unary()
            if exponent.free_symbols or not exponent.is_Integer:
                raise _MathInputError("unsupported_math_form")
            exp_value = int(exponent)
            if abs(exp_value) > MAX_POWER_ABS:
                raise _MathInputError("unsupported_math_form")
            if base == 0 and exp_value < 0:
                raise _MathInputError("division_by_zero")
            if exp_value < 0 and base.has(X):
                raise _MathInputError("variable_denominator_unsupported")
            return sp.Pow(base, exponent, evaluate=True)
        return base

    def _parse_atom(self) -> sp.Expr:
        tok = self._take()
        if tok.kind == "NUMBER":
            raw = tok.value.replace(",", ".")
            try:
                return sp.Rational(raw)
            except Exception as exc:
                raise _MathInputError("unsupported_math_form") from exc
        if tok.kind == "MIXED":
            match = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", tok.value)
            if not match:
                raise _MathInputError("unsupported_math_form")
            whole, numerator, denominator = map(int, match.groups())
            if denominator == 0:
                raise _MathInputError("division_by_zero")
            return sp.Rational(whole) + sp.Rational(numerator, denominator)
        if tok.kind == "VAR":
            return X
        if tok.kind == "LPAREN":
            value = self._parse_expression()
            close = self._peek()
            if close is None or close.kind != "RPAREN":
                raise _MathInputError("unsupported_math_form")
            self._take()
            return value
        raise _MathInputError("unsupported_math_form")


def _parse_expr(text: str) -> sp.Expr:
    """Parse a preflight-approved expression and build SymPy objects directly."""
    return _Parser(text).parse()


def _safe_parse_expr(text: Any) -> sp.Expr:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    return _parse_expr(text)


def _parse_equation(text: Any) -> tuple[sp.Expr, sp.Expr]:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    if text.count("=") != 1:
        raise _MathInputError("invalid_equation")
    lhs_text, rhs_text = text.split("=", 1)
    if not lhs_text.strip() or not rhs_text.strip():
        raise _MathInputError("invalid_equation")
    lhs = _parse_expr(lhs_text)
    rhs = _parse_expr(rhs_text)
    return lhs, rhs


def _serialize_exact(expr: sp.Expr) -> str:
    expr = sp.simplify(expr)
    return sp.sstr(expr)


def _solution_set(lhs: sp.Expr, rhs: sp.Expr) -> sp.Set:
    difference = sp.simplify(lhs - rhs)
    if not difference.has(X):
        return sp.S.Reals if difference == 0 else sp.S.EmptySet
    result = sp.solveset(difference, X, domain=sp.S.Reals)
    if isinstance(result, (sp.FiniteSet, sp.EmptySet)) or result == sp.S.Reals:
        return result
    raise _MathInputError("unsupported_solution_set")


def _serialize_solution_set(solution_set: sp.Set) -> dict[str, Any]:
    if solution_set is sp.S.EmptySet or solution_set == sp.S.EmptySet:
        return {"kind": "empty"}
    if solution_set == sp.S.Reals:
        return {"kind": "all_reals"}
    if isinstance(solution_set, sp.FiniteSet):
        values = sorted(solution_set, key=default_sort_key)
        return {"kind": "finite", "values": [_serialize_exact(value) for value in values]}
    raise _MathInputError("unsupported_solution_set")


def _same_set(left: sp.Set, right: sp.Set) -> bool:
    return left == right


def _numeric_value(text: str) -> sp.Expr:
    value = _safe_parse_expr(text)
    if value.free_symbols or value.is_number is not True:
        raise _MathInputError("not_numeric_expression")
    value = sp.simplify(value)
    if value in (sp.nan, sp.oo, -sp.oo, sp.zoo) or value.is_finite is False or value.is_real is False:
        raise _MathInputError("unsupported_math_form")
    return value


def _answer_value(text: str) -> sp.Expr:
    """Parse a plain numeric answer, allowing a single x=... assignment only."""
    if not isinstance(text, str):
        raise _MathInputError("invalid_answer")
    candidate = text.strip()
    if candidate.count("=") == 1:
        lhs_text, rhs_text = candidate.split("=", 1)
        if lhs_text.strip().casefold() != "x":
            raise _MathInputError("invalid_answer")
        candidate = rhs_text.strip()
    return _numeric_value(candidate)


def _parse_rounding_places(instruction: str | None) -> int | None:
    if not instruction:
        return None
    match = re.search(r"\brond\s+af\s+op\s+(\d+)\s+decimalen\b", instruction, re.IGNORECASE)
    if not match:
        # Do not silently ignore a rounding request whose precision cannot be
        # interpreted. Ordinary non-rounding task instructions remain neutral.
        if re.search(r"\brond\s+af\b|\bafronden\b", instruction, re.IGNORECASE):
            raise _MathInputError("rounding_instruction_not_supported")
        return None
    places = int(match.group(1))
    if places < 0 or places > 8:
        raise _MathInputError("rounding_instruction_not_supported")
    return places


def _round_decimal(value: sp.Expr, places: int) -> Decimal:
    """Round a real expression using decimal ROUND_HALF_UP semantics."""
    try:
        with localcontext() as ctx:
            ctx.prec = max(50, places + 30)
            if value.is_Rational:
                decimal_value = Decimal(int(value.p)) / Decimal(int(value.q))
            else:
                decimal_value = Decimal(str(sp.N(value, ctx.prec)))
            quantum = Decimal("1") if places == 0 else Decimal("1." + ("0" * places))
            return decimal_value.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise _MathInputError("unsupported_math_form") from exc


def _rounded_rational(value: sp.Expr, places: int) -> sp.Expr:
    """Return an exact rational representation of the rounded decimal."""
    rounded = _round_decimal(value, places)
    return sp.Rational(str(rounded))


def _rounded_display(value: sp.Expr, places: int) -> str:
    """Format the rounded decimal in Dutch notation, preserving requested places."""
    rounded = _round_decimal(value, places)
    text = f"{rounded:.{places}f}"
    return text.replace(".", ",")


def _result(status: str, check_type: str, input_value: Any, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "check_type": check_type,
        "input": input_value,
    }
    result.update(fields)
    return result


def _unverifiable(check_type: str, input_value: Any, error: _MathInputError, *, parser_invoked: bool = False) -> dict[str, Any]:
    return _result(
        "unverifiable", check_type, input_value,
        reason=error.reason,
        explanation_nl=error.message_nl,
        parser_invoked=parser_invoked,
    )


_UNDEFINED_PHRASES = {"kan niet", "bestaat niet", "niet mogelijk", "ongedefinieerd", "niet gedefinieerd"}


def _is_undefined_phrase(answer: str | None) -> bool:
    """True when the answer states that the result does not exist (e.g. 'kan niet')."""
    normalised = " ".join((answer or "").strip().casefold().split()).replace("ë", "e")
    return normalised in _UNDEFINED_PHRASES


def _verify_arithmetic(check_type: str, input_value: str, submitted_answer: str | None) -> dict[str, Any]:
    ok, reason = _preflight(input_value)
    if not ok:
        return _unverifiable(check_type, input_value, _MathInputError(reason or "disallowed_characters_or_syntax"))
    try:
        value = _numeric_value(input_value)
    except _MathInputError as exc:
        if exc.reason == "division_by_zero":
            correctly_identifies_undefined = _is_undefined_phrase(submitted_answer)
            return _result(
                "verified" if correctly_identifies_undefined else "wrong",
                check_type, input_value, reason=exc.reason,
                submitted_answer=submitted_answer,
                explanation_nl=("Delen door nul is niet gedefinieerd; het antwoord benoemt dit correct."
                                if correctly_identifies_undefined else exc.message_nl),
                parser_invoked=True,
            )
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    fields: dict[str, Any] = {"exact_value": _serialize_exact(value), "parser_invoked": True}
    if submitted_answer is None:
        return _result("verified", check_type, input_value, **fields,
                       explanation_nl="De uitdrukking is exact berekend.")
    if _is_undefined_phrase(submitted_answer):
        # The expression has a value, so "kan niet" / "bestaat niet" is a wrong answer.
        return _result(
            "wrong", check_type, input_value, **fields,
            submitted_answer=submitted_answer,
            explanation_nl=f"De uitdrukking heeft wel een waarde: {_serialize_exact(value)}.",
        )
    try:
        answer = _answer_value(submitted_answer)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct = sp.simplify(value - answer) == 0
    return _result(
        "verified" if correct else "wrong", check_type, input_value, **fields,
        submitted_answer=submitted_answer,
        explanation_nl=("Het antwoord komt overeen met de exacte waarde." if correct
                        else f"Het juiste exacte resultaat is {_serialize_exact(value)}."),
    )


def _verify_answer_equivalence(input_value: str, submitted_answer: str | None) -> dict[str, Any]:
    if submitted_answer is None:
        return _unverifiable("answer_equivalence", input_value, _MathInputError("invalid_answer"))
    ok, reason = _preflight(input_value)
    if not ok:
        return _unverifiable("answer_equivalence", input_value, _MathInputError(reason or "disallowed_characters_or_syntax"))
    try:
        input_value_exact = _numeric_value(input_value)
        answer = _answer_value(submitted_answer)
    except _MathInputError as exc:
        return _unverifiable("answer_equivalence", input_value, exc, parser_invoked=True)
    correct = sp.simplify(input_value_exact - answer) == 0
    return _result(
        "verified" if correct else "wrong", "answer_equivalence", input_value,
        exact_value=_serialize_exact(input_value_exact), submitted_answer=submitted_answer,
        parser_invoked=True,
        explanation_nl=("Beide notaties hebben dezelfde exacte waarde." if correct
                        else f"De exacte waarde is {_serialize_exact(input_value_exact)}."),
    )


def _verify_expression_equivalence(input_value: str) -> dict[str, Any]:
    try:
        lhs, rhs = _parse_equation(input_value)
        difference = sp.simplify(lhs - rhs)
        equivalent = difference == 0
    except _MathInputError as exc:
        return _unverifiable("expression_equivalence", input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    return _result(
        "verified" if equivalent else "wrong", "expression_equivalence", input_value,
        equivalent=bool(equivalent), parser_invoked=True,
        explanation_nl=("De twee uitdrukkingen zijn equivalent." if equivalent
                        else "De twee uitdrukkingen zijn niet voor alle toegestane x-waarden gelijk."),
    )


def _verify_equation_solution(input_value: str, submitted_answer: str | None, task_instruction: str | None) -> dict[str, Any]:
    try:
        lhs, rhs = _parse_equation(input_value)
        solutions = _solution_set(lhs, rhs)
        serialized_solutions = _serialize_solution_set(solutions)
    except _MathInputError as exc:
        return _unverifiable("equation_solution", input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)

    if submitted_answer is None:
        return _unverifiable("equation_solution", input_value, _MathInputError("invalid_answer"), parser_invoked=True)

    normalised_answer = " ".join(submitted_answer.strip().casefold().split())
    normalised_answer = normalised_answer.replace("ë", "e")
    if normalised_answer in {"geen oplossing", "geen oplossingen"}:
        correct = solutions == sp.S.EmptySet
        return _result("verified" if correct else "wrong", "equation_solution", input_value,
                       solution_set=serialized_solutions, submitted_answer=submitted_answer,
                       parser_invoked=True,
                       explanation_nl=("De vergelijking heeft geen reële oplossing." if correct
                                       else "De vergelijking heeft wel reële oplossingen."))
    if normalised_answer in {"alle getallen", "alle reele getallen", "alle reele waarden"}:
        correct = solutions == sp.S.Reals
        return _result("verified" if correct else "wrong", "equation_solution", input_value,
                       solution_set=serialized_solutions, submitted_answer=submitted_answer,
                       parser_invoked=True,
                       explanation_nl=("De vergelijking is waar voor alle reële getallen." if correct
                                       else "De vergelijking is niet waar voor alle reële getallen."))

    try:
        candidate = _answer_value(submitted_answer)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable("equation_solution", input_value, exc, parser_invoked=True)

    # A single supplied root counts as the complete answer only if the full
    # solution set is that singleton. A valid root is not enough when another
    # real root has been omitted.
    exact_match = isinstance(solutions, sp.FiniteSet) and len(solutions) == 1 and sp.simplify(candidate - next(iter(solutions))) == 0
    rounded_match = False
    rounded_answer: str | None = None
    if places is not None and isinstance(solutions, sp.FiniteSet) and len(solutions) == 1:
        exact_solution = next(iter(solutions))
        rounded_value = _rounded_rational(exact_solution, places)
        rounded_match = sp.simplify(candidate - rounded_value) == 0
        rounded_answer = _rounded_display(exact_solution, places)
    correct = bool(exact_match or rounded_match)
    fields: dict[str, Any] = {
        "solution_set": serialized_solutions,
        "submitted_answer": submitted_answer,
        "parser_invoked": True,
        "explanation_nl": ("Het antwoord is correct voor de volledige reële oplossingsverzameling." if correct
                           else "Het gegeven antwoord komt niet overeen met de volledige exacte oplossingsverzameling."),
    }
    if places is not None:
        fields["decimal_places"] = places
        if rounded_answer is not None:
            fields["rounded_answer"] = rounded_answer
    return _result("verified" if correct else "wrong", "equation_solution", input_value, **fields)


def _is_solved_form(equation_text: str) -> bool:
    try:
        lhs, rhs = _parse_equation(equation_text)
    except _MathInputError:
        return False
    return (lhs == X and not rhs.has(X)) or (rhs == X and not lhs.has(X))


def _steps_from_input(input_value: str) -> list[str]:
    # Compatibility with earlier proposed calls using arrow-delimited input.
    # Split before preflight so '->' is never passed into the expression parser.
    if "->" in input_value:
        return [part.strip() for part in input_value.split("->")]
    if "→" in input_value:
        return [part.strip() for part in input_value.split("→")]
    return [input_value.strip()]


def _verify_steps(input_value: str, steps: Sequence[str] | None) -> dict[str, Any]:
    given_steps = list(steps) if steps is not None else _steps_from_input(input_value)
    if len(given_steps) < 1 or any(not isinstance(step, str) or not step.strip() for step in given_steps):
        return _unverifiable("step_validation", input_value, _MathInputError("invalid_equation"))

    parsed_sets: list[sp.Set] = []
    parser_invoked = False
    try:
        for step in given_steps:
            lhs, rhs = _parse_equation(step)
            parser_invoked = True
            parsed_sets.append(_solution_set(lhs, rhs))
    except _MathInputError as exc:
        # A policy/length rejection can happen before parser entry; do not
        # imply the expression was analysed if it was not.
        return _unverifiable("step_validation", input_value, exc,
                             parser_invoked=parser_invoked and exc.reason not in _PREFLIGHT_REASONS)

    original_set = parsed_sets[0]
    failed_step: int | None = None
    for index in range(1, len(parsed_sets)):
        if not _same_set(parsed_sets[index - 1], parsed_sets[index]):
            failed_step = index + 1  # One-based numbering; step 1 is the original.
            break

    final_solved = _is_solved_form(given_steps[-1])
    fields: dict[str, Any] = {
        "failed_step": failed_step,
        "is_solved_form": final_solved,
        "original_solution_set": _serialize_solution_set(original_set),
        "parser_invoked": True,
    }
    if failed_step is None:
        status = "verified"
        fields["explanation_nl"] = ("Alle gecontroleerde stappen behouden dezelfde oplossingsverzameling." if final_solved
                                     else "Alle gecontroleerde stappen zijn equivalent; de vergelijking is nog niet opgelost in de vorm x = ... .")
    else:
        status = "wrong"
        fields["failed_step_solution_set"] = _serialize_solution_set(parsed_sets[failed_step - 1])
        fields["explanation_nl"] = f"De eerste ongeldige stap is stap {failed_step}: de oplossingsverzameling verandert ten opzichte van de vorige stap."
        consistent_after: list[int] = []
        for index in range(failed_step, len(parsed_sets)):
            if _same_set(parsed_sets[index - 1], parsed_sets[index]):
                consistent_after.append(index + 1)
            else:
                # Later consistency is measured from each local pair. If a
                # second error occurs, it is not labelled as carried forward.
                break
        if consistent_after:
            fields["steps_consistent_after_error"] = consistent_after
    return _result(status, "step_validation", input_value, **fields)


def verify_math(
    check_type: str,
    input: str = "",
    submitted_answer: str | None = None,
    steps: Sequence[str] | None = None,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Verify one supported mathematical claim and return a structured result.

    Args:
        check_type: one of arithmetic, answer_equivalence, expression_equivalence,
            equation_solution, step_validation.
        input: expression or equation; not used to infer a word problem's meaning.
        submitted_answer: answer to compare for arithmetic/equation checks.
        steps: explicit equation steps for step_validation. When omitted, an
            arrow-delimited input string is accepted for compatibility.
        task_instruction: optional Dutch rounding instruction, e.g.
            ``Rond af op 2 decimalen.``

    Returns a dictionary with status in {verified, wrong, unverifiable}.
    """
    allowed_types = {"arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution", "step_validation"}
    if check_type not in allowed_types:
        return _unverifiable(str(check_type), input, _MathInputError("unsupported_math_form"))
    if not isinstance(input, str):
        return _unverifiable(check_type, input, _MathInputError("disallowed_characters_or_syntax"))

    if check_type == "arithmetic":
        return _verify_arithmetic(check_type, input, submitted_answer)
    if check_type == "answer_equivalence":
        return _verify_answer_equivalence(input, submitted_answer)
    if check_type == "expression_equivalence":
        return _verify_expression_equivalence(input)
    if check_type == "equation_solution":
        return _verify_equation_solution(input, submitted_answer, task_instruction)
    return _verify_steps(input, steps)


if __name__ == "__main__":
    # Tiny CLI for manual smoke checks; not a production MCP transport.
    import argparse
    import json

    cli = argparse.ArgumentParser(description="Conservative Dutch maths verifier")
    cli.add_argument("check_type", choices=["arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution", "step_validation"])
    cli.add_argument("input", help="Expression, equation, or arrow-delimited steps")
    cli.add_argument("--answer")
    cli.add_argument("--instruction")
    args = cli.parse_args()
    print(json.dumps(verify_math(args.check_type, args.input, args.answer, task_instruction=args.instruction), ensure_ascii=False, indent=2))
