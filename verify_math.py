"""A small, conservative mathematics verifier for Dutch classroom notation.

The module intentionally does NOT use sympy.parse_expr or Python eval. A tiny
recursive-descent parser accepts a restricted mathematical grammar and builds
SymPy objects directly. Unsupported or ambiguous input fails closed as
'unverifiable'.

Scope:
- exact numeric arithmetic and answer-equivalence checks;
- symbolic identity checks with single-letter variables;
- real solution-set checks and step-by-step checks for equations;
- letter formulas: substituting values and rewriting for another letter,
  with nonzero conditions from letters in a denominator;
- metric unit conversions from a fixed unit table (including Dutch km/u).

This is a verification aid, not a proof of pedagogical correctness or of the
semantic modelling of a word problem. See README.md for the limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re
from typing import Any, Sequence

import sympy as sp
from sympy.core.sorting import default_sort_key


__all__ = ["verify_math", "MAX_INPUT_LENGTH", "MAX_TOKENS"]

MAX_INPUT_LENGTH = 256
MAX_TOKENS = 128
MAX_NUMERIC_DIGITS = 32
MAX_POWER_ABS = 20
_SYMBOLS: dict[str, sp.Symbol] = {}


def _symbol(name: str) -> sp.Symbol:
    """One real SymPy symbol per single ASCII letter; letters are case-sensitive."""
    if name not in _SYMBOLS:
        _SYMBOLS[name] = sp.Symbol(name, real=True)
    return _SYMBOLS[name]


X = _symbol("x")

# e and i are ordinary variables in basic algebra. As soon as e is raised to
# any power (e², e^x: could be Euler's number) or i to a power of 2 or more
# (i²: could be the imaginary unit), the tool refuses to guess and returns
# 'ambiguous_constant'. π is always the constant pi.
_RESERVED_LETTERS: frozenset[str] = frozenset()
_EULER = "e"
_IMAGINARY = "i"
_SINGLE_LETTER_RE = re.compile(r"[A-Za-z]")
# Three or more adjacent letters are a word (woordformule or prose), never a
# product: 'ab' means a·b, but 'aantal' is not a·a·n·t·a·l.
_LETTER_RUN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")

# Keep the allowlist intentionally small. Single ASCII letters are variables;
# Dutch prose is classified separately as a semantic formalisation task.
# Quotes, underscores and other executable-looking syntax are rejected before
# the parser is invoked.
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-zπ+\-−–—*/:×·()^²³=,\.\s]*$")
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
    "variable_denominator_unsupported": "Een noemer met een letter kan domeinbeperkingen geven; deze invoer wordt niet automatisch vereenvoudigd.",
    "reserved_symbol": "Dit symbool is gereserveerd en kan niet als variabele worden gebruikt.",
    "no_claim_to_check": "Er is geen antwoord van de leerling meegegeven; de waarde is berekend, maar er is niets gecontroleerd.",
    "unit_mismatch": "De eenheden horen niet bij dezelfde grootheid of worden niet ondersteund.",
    "ambiguous_constant": "Hier kan e het getal van Euler of i de imaginaire eenheid zijn. Geef aan of het een variabele of een constante is; de tool kiest niet zelf.",
    "word_variables_not_supported": "Woordformules met namen van meerdere letters worden nog niet ondersteund; gebruik één letter per variabele.",
    "ambiguous_unknown": "De vergelijking bevat meerdere letters; geef aan voor welke letter moet worden opgelost.",
    "invalid_variable": "De opgegeven onbekende is geen geldige letter uit de vergelijking.",
    "invalid_substitution": "De in te vullen waarden moeten bij letters uit de formule horen en getallen zijn.",
    "unsupported_variable": "Deze variabele wordt niet ondersteund.",
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
    "reserved_symbol",
    "word_variables_not_supported",
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

    words = _LETTER_RUN_RE.findall(text)
    if words:
        # A compact formula with a few multi-letter names is a woordformule;
        # anything else with words is prose that a human/model must first turn
        # into a formula. This function must not invent an equation from it.
        if "=" in text and not re.search(r"[.!?]", text) and len(words) <= 3 and _ALLOWED_CHARS.fullmatch(text):
            return False, "word_variables_not_supported"
        return False, "requires_semantic_formalisation"

    if not _ALLOWED_CHARS.fullmatch(text):
        if _WORD_RE.search(text):
            return False, "requires_semantic_formalisation"
        return False, "disallowed_characters_or_syntax"

    # Only single ASCII letters remain, and each one is a variable.
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

        if char == "π":
            tokens.append(_Token("CONST", "pi", pos, pos + 1))
            pos += 1
        elif char.isascii() and char.isalpha():
            if char in _RESERVED_LETTERS:
                raise _MathInputError("reserved_symbol")
            tokens.append(_Token("VAR", char, pos, pos + 1))
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
    """Recursive-descent parser for arithmetic and expressions with letters."""
    def __init__(self, text: str):
        self.tokens = _tokenize(text)
        self.index = 0
        self.denominators: list[sp.Expr] = []

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
        return token is not None and token.kind in ("VAR", "CONST", "LPAREN")

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
                    # SymPy cancels x/x to 1 while building the expression, so
                    # every denominator with a letter is remembered separately.
                    if rhs.free_symbols:
                        self.denominators.append(rhs)
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
            # Any power of e (e², e^x) may be Euler's number and i² the
            # imaginary unit: ask for context instead of silently treating
            # them as variables.
            if base == _symbol(_EULER):
                raise _MathInputError("ambiguous_constant")
            if base == _symbol(_IMAGINARY) and exponent.is_Integer and int(exponent) >= 2:
                raise _MathInputError("ambiguous_constant")
            if exponent.free_symbols or not exponent.is_Integer:
                raise _MathInputError("unsupported_math_form")
            exp_value = int(exponent)
            if abs(exp_value) > MAX_POWER_ABS:
                raise _MathInputError("unsupported_math_form")
            if base == 0 and exp_value < 0:
                raise _MathInputError("division_by_zero")
            if exp_value < 0 and base.free_symbols:
                self.denominators.append(base)
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
            return _symbol(tok.value)
        if tok.kind == "CONST":
            return sp.pi
        if tok.kind == "LPAREN":
            value = self._parse_expression()
            close = self._peek()
            if close is None or close.kind != "RPAREN":
                raise _MathInputError("unsupported_math_form")
            self._take()
            return value
        raise _MathInputError("unsupported_math_form")


def _parse_expr(text: str, denominators: list[sp.Expr] | None = None) -> sp.Expr:
    """Parse an approved expression and optionally retain its domain restrictions."""
    parser = _Parser(text)
    result = parser.parse()
    if denominators is not None:
        denominators.extend(parser.denominators)
    return result


def _safe_parse_expr(text: Any, allow_variable_denominators: bool = False) -> sp.Expr:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    denominators: list[sp.Expr] = []
    value = _parse_expr(text, denominators)
    if denominators and not allow_variable_denominators:
        raise _MathInputError("variable_denominator_unsupported")
    return value


def _parse_equation(text: Any, denominators: list[sp.Expr] | None = None) -> tuple[sp.Expr, sp.Expr]:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    if text.count("=") != 1:
        raise _MathInputError("invalid_equation")
    lhs_text, rhs_text = text.split("=", 1)
    if not lhs_text.strip() or not rhs_text.strip():
        raise _MathInputError("invalid_equation")
    lhs = _parse_expr(lhs_text, denominators)
    rhs = _parse_expr(rhs_text, denominators)
    return lhs, rhs


def _domain_restrictions(expressions: Sequence[sp.Expr]) -> list[str]:
    """Readable nonzero conditions, without duplicates or redundant factors.

    For a quotient only the numerator matters (the denominator is listed as
    its own condition), and a constant factor never changes 'is it zero', so
    s/v ≠ 0 becomes s ≠ 0 and 2x ≠ 0 becomes x ≠ 0.
    """
    result: list[str] = []
    seen: set[str] = set()
    for expression in expressions:
        numerator, _ = sp.together(expression).as_numer_denom()
        _, core = sp.factor(numerator).as_coeff_Mul()
        if not core.free_symbols:
            continue  # a nonzero constant is always nonzero
        condition = f"{sp.sstr(core)} ≠ 0"
        if condition not in seen:
            seen.add(condition)
            result.append(condition)
    return result


def _serialize_exact(expr: sp.Expr) -> str:
    expr = sp.simplify(expr)
    return sp.sstr(expr)


def _solve_linear(difference: sp.Expr, unknown: sp.Symbol) -> sp.Expr:
    """Solve difference = 0 for unknown when it is linear with a numeric coefficient.

    Used when other letters are present (formulas). A coefficient containing a
    letter would mean dividing by that letter, which needs a domain condition,
    so it fails closed.
    """
    try:
        poly = sp.Poly(sp.expand(difference), unknown)
    except sp.PolynomialError as exc:
        raise _MathInputError("unsupported_solution_set") from exc
    if poly.degree() != 1:
        raise _MathInputError("unsupported_solution_set")
    coefficient, constant = poly.all_coeffs()
    if coefficient.free_symbols:
        raise _MathInputError("variable_denominator_unsupported")
    return sp.simplify(-constant / coefficient)


def _pick_unknown(symbols: set, variable: str | None) -> sp.Symbol:
    """Choose the letter to solve for; never guess when there are several."""
    if variable is not None:
        if (not isinstance(variable, str) or not _SINGLE_LETTER_RE.fullmatch(variable)
                or variable in _RESERVED_LETTERS):
            raise _MathInputError("invalid_variable")
        unknown = _symbol(variable)
        if symbols and unknown not in symbols:
            raise _MathInputError("invalid_variable")
        return unknown
    if len(symbols) == 1:
        return next(iter(symbols))
    if not symbols:
        return X
    raise _MathInputError("ambiguous_unknown")


def _solution_set(lhs: sp.Expr, rhs: sp.Expr, unknown: sp.Symbol = X) -> sp.Set:
    difference = sp.simplify(lhs - rhs)
    others = difference.free_symbols - {unknown}
    if not difference.has(unknown):
        if others:
            raise _MathInputError("unsupported_solution_set")
        return sp.S.Reals if difference == 0 else sp.S.EmptySet
    if others:
        return sp.FiniteSet(_solve_linear(difference, unknown))
    result = sp.solveset(difference, unknown, domain=sp.S.Reals)
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


def _parse_assignment(text: Any, allow_variable_denominators: bool = False) -> tuple[str | None, sp.Expr]:
    """Parse an answer such as 'p = 3', 'a = 10 - 2b' or just '3'.

    Returns the letter on the left (or None) and the expression on the right.
    """
    if not isinstance(text, str):
        raise _MathInputError("invalid_answer")
    candidate = text.strip()
    name: str | None = None
    if candidate.count("=") == 1:
        lhs_text, rhs_text = candidate.split("=", 1)
        lhs_text = lhs_text.strip()
        if not _SINGLE_LETTER_RE.fullmatch(lhs_text):
            raise _MathInputError("invalid_answer")
        if lhs_text in _RESERVED_LETTERS:
            raise _MathInputError("reserved_symbol")
        name = lhs_text
        candidate = rhs_text.strip()
    elif "=" in candidate:
        raise _MathInputError("invalid_answer")
    expr = _safe_parse_expr(candidate, allow_variable_denominators=allow_variable_denominators)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise _MathInputError("unsupported_math_form")
    return name, expr


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
            if submitted_answer is None:
                return _result("unverifiable", check_type, input_value, reason="no_claim_to_check",
                               explanation_nl="De uitdrukking bevat een deling door nul; er is geen antwoord om te controleren.",
                               parser_invoked=True)
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
        # Only a calculation happened: no student claim was checked, so this
        # must never be reported as 'verified'. The value stays available.
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=_REASON_NL["no_claim_to_check"])
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
    denominators: list[sp.Expr] = []
    try:
        lhs, rhs = _parse_equation(input_value, denominators)
        difference = sp.simplify(lhs - rhs)
        equivalent = difference == 0
    except _MathInputError as exc:
        return _unverifiable("expression_equivalence", input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    fields: dict[str, Any] = {"equivalent": bool(equivalent), "parser_invoked": True}
    if denominators:
        fields["domain_restrictions"] = _domain_restrictions(denominators)
    return _result(
        "verified" if equivalent else "wrong", "expression_equivalence", input_value,
        **fields,
        explanation_nl=("De twee uitdrukkingen zijn equivalent op het domein waar de noemers niet nul zijn."
                        if equivalent and denominators else
                        "De twee uitdrukkingen zijn equivalent." if equivalent else
                        "De twee uitdrukkingen zijn niet equivalent."),
    )


def _verify_equation_solution(input_value: str, submitted_answer: str | None, task_instruction: str | None,
                              variable: str | None = None) -> dict[str, Any]:
    try:
        denominators: list[sp.Expr] = []
        lhs, rhs = _parse_equation(input_value, denominators)
        if denominators:
            raise _MathInputError("variable_denominator_unsupported")
        unknown = _pick_unknown(lhs.free_symbols | rhs.free_symbols, variable)
        solutions = _solution_set(lhs, rhs, unknown)
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
        answer_name, candidate = _parse_assignment(submitted_answer)
        places = _parse_rounding_places(task_instruction)
        if candidate.has(unknown):
            raise _MathInputError("invalid_answer")
    except _MathInputError as exc:
        return _unverifiable("equation_solution", input_value, exc, parser_invoked=True)

    # A single supplied root counts as the complete answer only if the full
    # solution set is that singleton. A valid root is not enough when another
    # real root has been omitted.
    exact_match = isinstance(solutions, sp.FiniteSet) and len(solutions) == 1 and sp.simplify(candidate - next(iter(solutions))) == 0
    rounded_match = False
    rounded_answer: str | None = None
    if (places is not None and isinstance(solutions, sp.FiniteSet) and len(solutions) == 1
            and candidate.is_number and next(iter(solutions)).is_number):
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
    fields["variable"] = unknown.name
    if answer_name is not None and answer_name != unknown.name:
        # Decision 27 Oct 2026: the equation is solved for a specific letter,
        # so a correct value under another letter is 'wrong'; value_correct
        # tells the teacher the arithmetic itself was right.
        fields["variable_mismatch"] = True
        fields["value_correct"] = correct
        if correct:
            fields["explanation_nl"] = (f"De waarde klopt, maar de vergelijking wordt opgelost naar {unknown.name}; "
                                        f"schrijf {unknown.name} = ... in plaats van {answer_name} = ... .")
        correct = False
    return _result("verified" if correct else "wrong", "equation_solution", input_value, **fields)


def _verify_formula_substitution(input_value: str, substitutions: Any, submitted_answer: str | None) -> dict[str, Any]:
    """Fill in given values (e.g. p = 3 in w = 4p + 10) and check the result.

    The letter that is left over may also sit in a denominator, e.g. t in
    v = s / t with v and s given: the formula is then brought to one fraction
    and its numerator is solved, and the result is checked against every
    denominator.
    """
    check_type = "formula_substitution"
    denominators: list[sp.Expr] = []
    try:
        lhs, rhs = _parse_equation(input_value, denominators)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    try:
        if not isinstance(substitutions, dict) or not substitutions:
            raise _MathInputError("invalid_substitution")
        formula_symbols = lhs.free_symbols | rhs.free_symbols
        mapping: dict[sp.Symbol, sp.Expr] = {}
        for name, raw_value in substitutions.items():
            if (not isinstance(name, str) or not _SINGLE_LETTER_RE.fullmatch(name)
                    or name in _RESERVED_LETTERS or _symbol(name) not in formula_symbols):
                raise _MathInputError("invalid_substitution")
            mapping[_symbol(name)] = _numeric_value(str(raw_value))
        for denominator in denominators:
            if sp.simplify(denominator.subs(mapping)) == 0:
                raise _MathInputError("division_by_zero")
        filled_in = sp.together((lhs - rhs).subs(mapping))
        remaining = filled_in.free_symbols
        if len(remaining) != 1:
            raise _MathInputError("ambiguous_unknown")
        target = next(iter(remaining))
        numerator, fraction_denominator = filled_in.as_numer_denom()
        value = _solve_linear(numerator, target)
        if not value.is_number:
            raise _MathInputError("unsupported_solution_set")
        for denominator in [*denominators, fraction_denominator]:
            evaluated_denominator = sp.simplify(denominator.subs(mapping).subs({target: value}))
            if evaluated_denominator == 0:
                raise _MathInputError("division_by_zero")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)

    fields: dict[str, Any] = {
        "exact_value": _serialize_exact(value),
        "target_variable": target.name,
        "parser_invoked": True,
    }
    if denominators:
        fields["domain_restrictions"] = _domain_restrictions(denominators)
    if submitted_answer is None:
        # Same rule as arithmetic: a value without a claim is not 'verified'.
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"Na invullen is {target.name} = {_serialize_exact(value)}, "
                                      "maar er is geen antwoord van de leerling meegegeven.")
    try:
        answer_name, candidate = _parse_assignment(submitted_answer)
        if candidate.free_symbols:
            raise _MathInputError("invalid_answer")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct = sp.simplify(candidate - value) == 0
    fields["submitted_answer"] = submitted_answer
    fields["explanation_nl"] = ("Het antwoord klopt na invullen." if correct
                                else f"Na invullen is {target.name} = {_serialize_exact(value)}.")
    if answer_name is not None and answer_name != target.name:
        # Same rule as equation_solution: right value, wrong letter is 'wrong'.
        fields["variable_mismatch"] = True
        fields["value_correct"] = correct
        if correct:
            fields["explanation_nl"] = (f"De waarde klopt, maar de gevraagde letter is {target.name}, "
                                        f"niet {answer_name}.")
        correct = False
    return _result("verified" if correct else "wrong", check_type, input_value, **fields)


def _verify_formula_rewrite(input_value: str, submitted_answer: str | None) -> dict[str, Any]:
    """Check a formula rewrite and retain any conditions introduced by denominators."""
    check_type = "formula_rewrite"
    denominators: list[sp.Expr] = []
    try:
        lhs, rhs = _parse_equation(input_value, denominators)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    try:
        if submitted_answer is None:
            raise _MathInputError("invalid_answer")
        answer_name, candidate = _parse_assignment(submitted_answer, allow_variable_denominators=True)
        if answer_name is None:
            raise _MathInputError("invalid_answer")
        target = _symbol(answer_name)
        if target not in (lhs.free_symbols | rhs.free_symbols) or candidate.has(target):
            raise _MathInputError("invalid_answer")
        difference = sp.together(lhs - rhs)
        numerator, _ = difference.as_numer_denom()
        poly = sp.Poly(sp.expand(numerator), target)
        if poly.degree() != 1:
            raise _MathInputError("unsupported_solution_set")
        coefficient, constant = poly.all_coeffs()
        if coefficient == 0:
            raise _MathInputError("unsupported_solution_set")
        solution = sp.simplify(-constant / coefficient)
        correct = sp.simplify(candidate - solution) == 0
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    except sp.PolynomialError:
        return _unverifiable(check_type, input_value, _MathInputError("unsupported_solution_set"),
                             parser_invoked=True)
    conditions: list[sp.Expr] = []
    for denominator in denominators:
        condition = sp.simplify(denominator.subs({target: solution}))
        if condition == 0:
            return _unverifiable(check_type, input_value, _MathInputError("unsupported_solution_set"),
                                 parser_invoked=True)
        conditions.append(condition)
    coefficient_condition = sp.simplify(coefficient)
    if coefficient_condition.free_symbols:
        conditions.append(coefficient_condition)
    if solution.is_rational_function(*solution.free_symbols):
        _, solution_denominator = sp.together(solution).as_numer_denom()
        if solution_denominator != 1:
            conditions.append(solution_denominator)
    expected = f"{target.name} = {sp.sstr(solution)}"
    fields: dict[str, Any] = {
        "target_variable": target.name, "expected_formula": expected,
        "submitted_answer": submitted_answer, "parser_invoked": True,
    }
    if conditions:
        fields["conditions"] = _domain_restrictions(conditions)
    return _result(
        "verified" if correct else "wrong", check_type, input_value,
        **fields,
        explanation_nl=("De herschreven formule klopt onder de genoemde voorwaarden."
                        if correct and conditions else
                        "De herschreven formule is equivalent met de oorspronkelijke." if correct else
                        f"Herschreven naar {target.name} hoort de formule {expected} te zijn."),
    )


def _is_solved_form(equation_text: str, unknown: sp.Symbol = X) -> bool:
    try:
        lhs, rhs = _parse_equation(equation_text)
    except _MathInputError:
        return False
    return (lhs == unknown and not rhs.has(unknown)) or (rhs == unknown and not lhs.has(unknown))


def _steps_from_input(input_value: str) -> list[str]:
    # Compatibility with earlier proposed calls using arrow-delimited input.
    # Split before preflight so '->' is never passed into the expression parser.
    if "->" in input_value:
        return [part.strip() for part in input_value.split("->")]
    if "→" in input_value:
        return [part.strip() for part in input_value.split("→")]
    return [input_value.strip()]


def _verify_steps(input_value: str, steps: Sequence[str] | None, variable: str | None = None) -> dict[str, Any]:
    given_steps = list(steps) if steps is not None else _steps_from_input(input_value)
    if len(given_steps) < 1 or any(not isinstance(step, str) or not step.strip() for step in given_steps):
        return _unverifiable("step_validation", input_value, _MathInputError("invalid_equation"))

    parsed_sets: list[sp.Set] = []
    parser_invoked = False
    try:
        equations = []
        for step in given_steps:
            denominators: list[sp.Expr] = []
            equations.append(_parse_equation(step, denominators))
            parser_invoked = True
            if denominators:
                raise _MathInputError("variable_denominator_unsupported")
        all_symbols: set = set()
        for lhs, rhs in equations:
            all_symbols |= lhs.free_symbols | rhs.free_symbols
        unknown = _pick_unknown(all_symbols, variable)
        for lhs, rhs in equations:
            parsed_sets.append(_solution_set(lhs, rhs, unknown))
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

    final_solved = _is_solved_form(given_steps[-1], unknown)
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
        # Every later step is judged against the step directly before it, so a
        # student who makes a second, independent error gets both errors
        # reported: invalid steps in additional_invalid_steps, and steps that
        # correctly carry an earlier error forward in steps_consistent_after_error.
        consistent_after: list[int] = []
        additional_invalid: list[int] = []
        for index in range(failed_step, len(parsed_sets)):
            if _same_set(parsed_sets[index - 1], parsed_sets[index]):
                consistent_after.append(index + 1)
            else:
                additional_invalid.append(index + 1)
        if consistent_after:
            fields["steps_consistent_after_error"] = consistent_after
        if additional_invalid:
            fields["additional_invalid_steps"] = additional_invalid
            listed = ", ".join(str(step) for step in additional_invalid)
            fields["explanation_nl"] += f" Daarna is ook stap {listed} ongeldig ten opzichte van de vorige stap."
    # Only meaningful when the last line has the form x = ...: does the final
    # answer match the solution set of the original equation?
    fields["final_answer_correct"] = _same_set(parsed_sets[-1], original_set) if final_solved else None
    return _result(status, "step_validation", input_value, **fields)


def _unit_entries() -> dict[str, tuple[str, sp.Rational]]:
    """Fixed metric unit table: unit -> (dimension, factor to the base unit)."""
    r = sp.Rational
    table: dict[str, tuple[str, sp.Rational]] = {
        # Length (base: metre)
        "mm": ("length", r(1, 1000)), "cm": ("length", r(1, 100)), "dm": ("length", r(1, 10)),
        "m": ("length", r(1)), "dam": ("length", r(10)), "hm": ("length", r(100)), "km": ("length", r(1000)),
        # Mass (base: gram)
        "mg": ("mass", r(1, 1000)), "cg": ("mass", r(1, 100)), "dg": ("mass", r(1, 10)), "g": ("mass", r(1)),
        "dag": ("mass", r(10)), "hg": ("mass", r(100)), "kg": ("mass", r(1000)), "t": ("mass", r(1000000)),
        # Time (base: second). Dutch schools write 'u' for uur; 'h' is accepted too.
        "s": ("time", r(1)), "min": ("time", r(60)), "u": ("time", r(3600)), "h": ("time", r(3600)),
        "d": ("time", r(86400)),
        # Area (base: square metre)
        "mm²": ("area", r(1, 1000000)), "cm²": ("area", r(1, 10000)), "dm²": ("area", r(1, 100)),
        "m²": ("area", r(1)), "km²": ("area", r(1000000)), "ha": ("area", r(10000)),
        # Volume (base: cubic metre). Litres are volume too, so 1 dm³ = 1 L can be checked.
        "mm³": ("volume", r(1, 10**9)), "cm³": ("volume", r(1, 10**6)), "dm³": ("volume", r(1, 1000)),
        "m³": ("volume", r(1)),
        # Speed (base: metres per second). Dutch: km/u.
        "m/s": ("speed", r(1)), "km/u": ("speed", r(5, 18)), "km/h": ("speed", r(5, 18)),
        "m/min": ("speed", r(1, 60)),
    }
    # Capacity units, in both spellings used in Dutch schools (ml and mL).
    litre_factors = {"m": r(1, 10**6), "c": r(1, 10**5), "d": r(1, 10**4), "": r(1, 1000),
                     "da": r(1, 100), "h": r(1, 10), "k": r(1)}
    for prefix, factor in litre_factors.items():
        table[f"{prefix}L"] = ("volume", factor)
        table[f"{prefix}l"] = ("volume", factor)
    return table


_UNIT_TABLE = _unit_entries()
_QUANTITY_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:,\d+)?|\d+\s*/\s*\d+))\s*"
    r"([A-Za-z]+(?:²|³|\^2|\^3)?(?:/[A-Za-z]+)?)\s*$"
)


def _parse_quantity(value: Any) -> tuple[sp.Expr, str, str]:
    if not isinstance(value, str):
        raise _MathInputError("unit_mismatch")
    match = _QUANTITY_RE.fullmatch(value)
    if not match:
        raise _MathInputError("unit_mismatch")
    number, unit = match.groups()
    unit = unit.replace("^2", "²").replace("^3", "³")
    if unit not in _UNIT_TABLE:
        raise _MathInputError("unit_mismatch")
    dimension, _ = _UNIT_TABLE[unit]
    return _numeric_value(number), unit, dimension


def _verify_unit_conversion(input_value: str, submitted_answer: str | None,
                            task_instruction: str | None = None) -> dict[str, Any]:
    check_type = "unit_conversion"
    try:
        input_value_number, input_unit, input_dimension = _parse_quantity(input_value)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=False)
    _, input_factor = _UNIT_TABLE[input_unit]
    if submitted_answer is None:
        return _result(
            "unverifiable", check_type, input_value, reason="no_claim_to_check",
            exact_value=_serialize_exact(input_value_number),
            source_unit=input_unit, explanation_nl="De hoeveelheid is gelezen, maar er is geen omrekening van de leerling om te controleren.",
        )
    try:
        answer_number, answer_unit, answer_dimension = _parse_quantity(submitted_answer)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    if input_dimension != answer_dimension:
        return _unverifiable(check_type, input_value, _MathInputError("unit_mismatch"), parser_invoked=True)
    _, answer_factor = _UNIT_TABLE[answer_unit]
    exact_value = sp.simplify(input_value_number * input_factor / answer_factor)
    exact_match = sp.simplify(exact_value - answer_number) == 0
    fields: dict[str, Any] = {
        "exact_value": _serialize_exact(exact_value), "target_unit": answer_unit,
        "source_unit": input_unit, "submitted_answer": submitted_answer, "parser_invoked": True,
    }
    # Same rounding policy as equations: an approximation only counts when the
    # task asks for it, e.g. 50 km/u = 13,89 m/s with "Rond af op 2 decimalen."
    rounded_match = False
    if places is not None:
        rounded_match = sp.simplify(answer_number - _rounded_rational(exact_value, places)) == 0
        fields["decimal_places"] = places
        fields["rounded_answer"] = _rounded_display(exact_value, places)
    correct = bool(exact_match or rounded_match)
    return _result(
        "verified" if correct else "wrong", check_type, input_value, **fields,
        explanation_nl=("De omrekening klopt exact." if exact_match else
                        "De omrekening klopt na afronden." if correct else
                        f"De hoeveelheid is {_serialize_exact(exact_value)} {answer_unit}."),
    )


def verify_math(
    check_type: str,
    input: str = "",
    submitted_answer: str | None = None,
    steps: Sequence[str] | None = None,
    task_instruction: str | None = None,
    variable: str | None = None,
    substitutions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify one supported mathematical claim and return a structured result.

    Args:
        check_type: one of arithmetic, answer_equivalence, expression_equivalence,
            equation_solution, step_validation, formula_substitution, formula_rewrite,
            unit_conversion.
        input: expression, equation or quantity; not used to infer a word problem's meaning.
        submitted_answer: the student's answer to compare, exactly as written.
        steps: explicit equation steps for step_validation. When omitted, an
            arrow-delimited input string is accepted for compatibility.
        task_instruction: optional Dutch rounding instruction, e.g.
            ``Rond af op 2 decimalen.`` (equation_solution, unit_conversion).
        variable: the letter to solve for when an equation contains more than
            one letter (equation_solution, step_validation).
        substitutions: values to fill in for formula_substitution, e.g.
            ``{"p": "3"}``.

    Returns a dictionary with status in {verified, wrong, unverifiable}.
    """
    allowed_types = {"arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution",
                     "step_validation", "formula_substitution", "formula_rewrite", "unit_conversion"}
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
        return _verify_equation_solution(input, submitted_answer, task_instruction, variable)
    if check_type == "formula_substitution":
        return _verify_formula_substitution(input, substitutions, submitted_answer)
    if check_type == "formula_rewrite":
        return _verify_formula_rewrite(input, submitted_answer)
    if check_type == "unit_conversion":
        return _verify_unit_conversion(input, submitted_answer, task_instruction)
    return _verify_steps(input, steps, variable)


if __name__ == "__main__":
    # Tiny CLI for manual smoke checks; not a production MCP transport.
    import argparse
    import json

    cli = argparse.ArgumentParser(description="Conservative Dutch maths verifier")
    cli.add_argument("check_type", choices=["arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution",
                                            "step_validation", "formula_substitution", "formula_rewrite", "unit_conversion"])
    cli.add_argument("input", help="Expression, equation, quantity, or arrow-delimited steps")
    cli.add_argument("--answer")
    cli.add_argument("--instruction")
    cli.add_argument("--variable")
    cli.add_argument("--set", action="append", default=[], metavar="LETTER=VALUE",
                     help="value to fill in for formula_substitution, e.g. --set p=3")
    args = cli.parse_args()
    values = dict(item.split("=", 1) for item in args.set) or None
    print(json.dumps(verify_math(args.check_type, args.input, args.answer, task_instruction=args.instruction,
                                 variable=args.variable, substitutions=values), ensure_ascii=False, indent=2))
