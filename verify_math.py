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
- metric unit conversions from a fixed unit table (including Dutch km/u);
- percentages: 25%, '25% van 80', and percentage change (80 -> 100 = 25%);
- linear inequalities in one letter, also double ones (-4 < 2x + 2 ≤ 8);
- systems of two linear equations in two letters (2x + y = 7; x - y = 2);
- function notation: f(x) = ... with a value f(-2) or an equation f(x) = g(x);
- statistics of a list: gemiddelde, mediaan, modus, spreidingsbreedte;
- scientific notation (a · 10^n with 1 ≤ |a| < 10) and superscript exponents;
- divisors, multiples and primes: ggd, kgv, priemfactoren, delers, priemgetal;
- counting: n!, n nCr k ('n boven k'), and choosing with or without order and replacement;
- geometry check types, forwarded to geometry.verify_geometry (angle sums, shape formulas,
  scale factor, sin/cos/tan, sine and cosine rule, distance and midpoint).

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
MAX_POWER_ABS = 30
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
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-zπ√%!+\-−–—*/:×·()^⁰¹²³⁴⁵⁶⁷⁸⁹⁻=,\.\s]*$")
# Superscript exponents (10⁴, 10⁻³, x⁴) are rewritten as ^(...) before tokenizing.
_SUPERSCRIPT_RE = re.compile(r"⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_SUPERSCRIPT_IN = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _expand_superscripts(text: str) -> str:
    return _SUPERSCRIPT_RE.sub(lambda match: "^(" + match.group(0).translate(_SUPERSCRIPT_IN) + ")", text)
_SUSPICIOUS_CHARS = set("_'\"[]{}@#$&`\\;")
# '10 boven 3' (Dutch school notation) is read as the calculator form 10 nCr 3.
_BOVEN_RE = re.compile(r"(?<=[\d)])\s*\bboven\b\s*(?=[\d(])", re.IGNORECASE)
_ALLOWED_WORDS = {"nCr"}
# 'p% van q' is the only place where the word 'van' is allowed: it means p% · q.
_PERCENT_OF_RE = re.compile(r"%\s*van\b", re.IGNORECASE)
# 80 + 15% means 92 on a calculator but 80,15 in algebra: never guess.
_AMBIGUOUS_PERCENT_RE = re.compile(
    r"(?<=[0-9A-Za-z)%π⁰¹²³⁴⁵⁶⁷⁸⁹])\s*[+\-−–]\s*\d+(?:,\d+)?\s*%"
    r"|\d\s*%\s*[+\-−–]\s*[0-9A-Za-z(π]"
)


def _normalize_percent_phrases(text: str) -> str:
    """Turn '25% van 80' into '25% · 80' and '10 boven 3' into '10 nCr 3' before any other check."""
    return _BOVEN_RE.sub(" nCr ", _PERCENT_OF_RE.sub("% ·", text))
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
    "not_real": "De wortel van een negatief getal is geen reëel getal.",
    "ambiguous_root_notation": "Bij een wortel met een macht erachter (√2²) is niet duidelijk wat bedoeld wordt; gebruik haakjes: √(2²) of (√2)².",
    "ambiguous_percentage": "Een getal plus of min een percentage (zoals 80 + 15%) is dubbelzinnig: 92 op een rekenmachine, 80,15 in de algebra. Schrijf 80 · 1,15 of 80 + 15% van 80.",
    "invalid_percentage_change": "Geef de oude en de nieuwe waarde als 'oud -> nieuw', bijvoorbeeld 80 -> 100, en het antwoord als percentage.",
    "invalid_inequality": "Voer één ongelijkheid in met <, >, ≤ of ≥, of een dubbele ongelijkheid zoals -4 < 2x + 2 ≤ 8.",
    "invalid_system": "Voer een stelsel in als twee vergelijkingen, gescheiden door ';' of een nieuwe regel, bijvoorbeeld 2x + y = 7; x - y = 2.",
    "invalid_function": "Geef eerst de functie(s), bijvoorbeeld f(x) = 2x + 1, en daarna de opdracht, gescheiden door ';': f(3) of f(x) = 7. Gebruik alleen functies die gedefinieerd zijn.",
    "ambiguous_counting": "Geef aan of de volgorde belangrijk is (met of zonder volgorde) en of er wordt teruggelegd (met of zonder terugleggen); de tool kiest dat niet zelf.",
    "invalid_counting_input": "Geef bijvoorbeeld 'met volgorde, zonder terugleggen: 10, 3': eerst het aantal mogelijkheden, dan het aantal keuzes. Zonder terugleggen kun je niet meer kiezen dan er zijn.",
    "unsupported_counting": "Zonder volgorde en met terugleggen wordt niet ondersteund.",
    "unsupported_number_task": "Deze opdracht wordt niet ondersteund; kies ggd, kgv, priemfactoren, delers of priemgetal.",
    "invalid_number_input": "Geef de opdracht en hele positieve getallen, bijvoorbeeld 'ggd: 12, 18' of 'priemfactoren: 360'.",
    "unsupported_statistic": "Deze statistische maat wordt (nog) niet ondersteund; kies gemiddelde, mediaan, modus of spreidingsbreedte.",
    "invalid_statistics_input": "Geef de maat en de getallen, bijvoorbeeld 'gemiddelde: 4, 7, 7, 9, 3' of 'mediaan: 1,5; 2,5; 3,5'.",
    "interval_notation_not_supported": "Intervalnotatie (zoals ⟨←, 3⟩ of [2, 5]) wordt nog niet gelezen; schrijf het antwoord als x < 3 of 2 ≤ x ≤ 5.",
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
    "ambiguous_percentage",
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
    text = _normalize_percent_phrases(text)
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

    if _AMBIGUOUS_PERCENT_RE.search(text):
        return False, "ambiguous_percentage"

    words = [word for word in _LETTER_RUN_RE.findall(text) if word not in _ALLOWED_WORDS]
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
    text = _normalize_percent_phrases(text)
    text = _expand_superscripts(text)
    text = text.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*", "·": "*"}))
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
        elif text.startswith("nCr", pos):
            tokens.append(_Token("OP", "nCr", pos, pos + 3))
            pos += 3
        elif char == "!":
            tokens.append(_Token("FACT", char, pos, pos + 1))
            pos += 1
        elif char.isascii() and char.isalpha():
            if char in _RESERVED_LETTERS:
                raise _MathInputError("reserved_symbol")
            tokens.append(_Token("VAR", char, pos, pos + 1))
            pos += 1
        elif char == "%":
            tokens.append(_Token("PCT", char, pos, pos + 1))
            pos += 1
        elif char == "√":
            tokens.append(_Token("SQRT", char, pos, pos + 1))
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
        # Bookkeeping for roots: conditions 'radicand ≥ 0' for letters, the
        # numbers under a root (None for a fraction) and whether a root ends up
        # in a denominator. Used for domain conditions and is_simplified.
        self.root_conditions: list[sp.Expr] = []
        self.radicands: list[int | None] = []
        self.root_in_denominator = False

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
        return token is not None and token.kind in ("VAR", "CONST", "LPAREN", "SQRT")

    def _parse_term(self) -> sp.Expr:
        result = self._parse_unary()
        while True:
            tok = self._peek()
            if tok is not None and tok.kind == "OP" and tok.value in ("*", "/"):
                op = self._take().value
                start = self.index
                rhs = self._parse_unary()
                if op == "/" and any(t.kind == "SQRT" for t in self.tokens[start:self.index]):
                    self.root_in_denominator = True
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
        result = self._parse_power_core()
        # Postfix percent and factorial bind like a power: 25% = 25/100,
        # 100 : 25% = 400, 2 · 3! = 12.
        while (tok := self._peek()) is not None and tok.kind in ("PCT", "FACT"):
            self._take()
            if tok.kind == "PCT":
                result = sp.Mul(result, sp.Rational(1, 100), evaluate=True)
            else:
                if not (result.is_Integer and 0 <= result <= 170):
                    raise _MathInputError("unsupported_math_form")
                result = sp.factorial(result)
        # n nCr k (also written 'n boven k'): number of combinations.
        while (tok := self._peek()) is not None and tok.kind == "OP" and tok.value == "nCr":
            self._take()
            k = self._parse_power()
            if not (result.is_Integer and k.is_Integer and 0 <= k and 0 <= result <= 1000):
                raise _MathInputError("unsupported_math_form")
            result = sp.binomial(result, k)
        return result

    def _parse_root(self) -> sp.Expr:
        """√ applies to the next number, letter, constant or bracket: √12 + 1 = (√12) + 1."""
        self._take()
        sign = 1
        nxt = self._peek()
        if nxt is not None and nxt.kind == "OP" and nxt.value in ("+", "-"):
            self._take()
            sign = -1 if nxt.value == "-" else 1
        operand = self._parse_atom()
        if sign == -1:
            operand = -operand
        if operand.is_number:
            if operand.is_negative:
                raise _MathInputError("not_real")
        elif operand.is_nonnegative is not True:
            # x² needs no condition; x or x + 1 must be ≥ 0.
            self.root_conditions.append(operand)
        if operand.is_Integer:
            self.radicands.append(int(operand))
        elif operand.is_Rational:
            self.radicands.append(None)  # a fraction under the root is not simplified
        return sp.sqrt(operand)

    def _parse_power_core(self) -> sp.Expr:
        tok = self._peek()
        if tok is not None and tok.kind == "SQRT":
            base = self._parse_root()
            after = self._peek()
            if after is not None and after.kind == "OP" and after.value == "^":
                raise _MathInputError("ambiguous_root_notation")
            return base
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


def _parse_expr(text: str, denominators: list[sp.Expr] | None = None,
                root_conditions: list[sp.Expr] | None = None) -> sp.Expr:
    """Parse an approved expression and optionally retain its domain restrictions."""
    parser = _Parser(text)
    result = parser.parse()
    if denominators is not None:
        denominators.extend(parser.denominators)
    if root_conditions is not None:
        root_conditions.extend(parser.root_conditions)
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


def _parse_equation(text: Any, denominators: list[sp.Expr] | None = None,
                    root_conditions: list[sp.Expr] | None = None) -> tuple[sp.Expr, sp.Expr]:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    if text.count("=") != 1:
        raise _MathInputError("invalid_equation")
    lhs_text, rhs_text = text.split("=", 1)
    if not lhs_text.strip() or not rhs_text.strip():
        raise _MathInputError("invalid_equation")
    lhs = _parse_expr(lhs_text, denominators, root_conditions)
    rhs = _parse_expr(rhs_text, denominators, root_conditions)
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


def _root_restrictions(radicands: Sequence[sp.Expr]) -> list[str]:
    """Conditions 'radicand ≥ 0' for roots with letters, without duplicates."""
    result: list[str] = []
    for radicand in radicands:
        condition = f"{sp.sstr(radicand)} ≥ 0"
        if condition not in result:
            result.append(condition)
    return result


def _is_simplified_root_form(text: Any) -> bool | None:
    """None when the answer has no √; otherwise whether its root form is simplified.

    Simplified: every number under a root is a whole number > 1 without a square
    factor (2√3, not √12) and no root is left in a denominator (√2/2, not 1/√2).
    """
    if not isinstance(text, str) or "√" not in text:
        return None
    candidate = text.split("=", 1)[1] if text.count("=") == 1 else text
    ok, _ = _preflight(candidate)
    if not ok:
        return None
    try:
        parser = _Parser(candidate)
        parser.parse()
    except _MathInputError:
        return None
    if parser.root_in_denominator:
        return False
    for radicand in parser.radicands:
        if radicand is None or radicand < 2:
            return False
        if any(power > 1 for power in sp.factorint(radicand).values()):
            return False
    return True


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
    # sp.EmptySet is a singleton object in recent SymPy, not a class: compare, don't isinstance.
    if isinstance(result, sp.FiniteSet) or result == sp.S.EmptySet or result == sp.S.Reals:
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


def _parse_answer_list(text: Any) -> list[tuple[str | None, sp.Expr]]:
    """Read 'x = √5 of x = -√5', 'x = ±√5' or a single 'x = 4'."""
    if not isinstance(text, str):
        raise _MathInputError("invalid_answer")
    parts = re.split(r"\s+of\s+|\s*∨\s*", text.strip(), flags=re.IGNORECASE)
    answers: list[tuple[str | None, sp.Expr]] = []
    for part in parts:
        if not part.strip():
            raise _MathInputError("invalid_answer")
        if "±" in part:
            if part.count("±") != 1:
                raise _MathInputError("invalid_answer")
            answers.append(_parse_assignment(part.replace("±", "+")))
            answers.append(_parse_assignment(part.replace("±", "-")))
        else:
            answers.append(_parse_assignment(part))
    return answers


def _matches_solution_set(candidates: Sequence[sp.Expr], solutions: sp.Set, places: int | None) -> bool:
    """Every solution given exactly once (or rounded, when rounding is asked for), in any order."""
    if not isinstance(solutions, sp.FiniteSet):
        return False
    remaining = list(solutions)
    if len(candidates) != len(remaining):
        return False
    for candidate in candidates:
        for index, solution in enumerate(remaining):
            if sp.simplify(candidate - solution) == 0:
                break
            if (places is not None and candidate.is_number and solution.is_number
                    and sp.simplify(candidate - _rounded_rational(solution, places)) == 0):
                break
        else:
            return False
        remaining.pop(index)
    return True


class _SigFigs(int):
    """A rounding request in significant figures; a plain int means decimal places."""


def _rounding_fields(places: int) -> dict[str, int]:
    if isinstance(places, _SigFigs):
        return {"significant_figures": int(places)}
    return {"decimal_places": int(places)}


def _parse_rounding_places(instruction: str | None) -> int | None:
    """Decimal places as an int, significant figures as _SigFigs, or None."""
    if not instruction:
        return None
    significant = re.search(r"\brond\s+af\s+op\s+(\d+)\s+significante?\s+cijfers?\b", instruction, re.IGNORECASE)
    if significant:
        figures = int(significant.group(1))
        if figures < 1 or figures > 15:
            raise _MathInputError("rounding_instruction_not_supported")
        return _SigFigs(figures)
    match = re.search(r"\brond\s+af\s+op\s+(\d+)\s+decima(?:al|len)\b", instruction, re.IGNORECASE)
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
            if isinstance(places, _SigFigs):
                return _round_significant(decimal_value, int(places))
            quantum = Decimal("1") if places == 0 else Decimal("1." + ("0" * places))
            return decimal_value.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise _MathInputError("unsupported_math_form") from exc


def _round_significant(value: Decimal, figures: int) -> Decimal:
    """Half up to significant figures: 2/3 -> 0.667, 12345 -> 12300, 9.996 -> 10.0."""
    if value == 0:
        return Decimal(0)
    exponent = value.adjusted()
    rounded = value.quantize(Decimal(1).scaleb(exponent - figures + 1), rounding=ROUND_HALF_UP)
    if rounded != 0 and rounded.adjusted() != exponent:
        # 9.996 -> 10.00 has one figure too many: round again at the new power of ten.
        rounded = value.quantize(Decimal(1).scaleb(rounded.adjusted() - figures + 1), rounding=ROUND_HALF_UP)
    return rounded


def _rounded_rational(value: sp.Expr, places: int) -> sp.Expr:
    """Return an exact rational representation of the rounded decimal."""
    rounded = _round_decimal(value, places)
    return sp.Rational(format(rounded, "f"))


def _rounded_display(value: sp.Expr, places: int) -> str:
    """Format the rounded decimal in Dutch notation, preserving requested places or figures."""
    rounded = _round_decimal(value, places)
    text = format(rounded, "f") if isinstance(places, _SigFigs) else f"{rounded:.{places}f}"
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


def _rounded_compare(exact: sp.Expr, answer: sp.Expr, places: int | None,
                     answer_text: str) -> tuple[bool, dict[str, Any]]:
    """Compare an answer with an exact value, allowing rounding only when asked.

    When the answer is written as a percentage (33,33%), rounding is applied to
    the percentage number, not to the fraction 0,3333.
    """
    if sp.simplify(exact - answer) == 0:
        return True, {}
    if places is None:
        return False, {}
    percent = answer_text.strip().endswith("%")
    scale = 100 if percent else 1
    rounded = _rounded_rational(exact * scale, places)
    display = _rounded_display(exact * scale, places) + ("%" if percent else "")
    matched = sp.simplify(answer * scale - rounded) == 0
    return matched, {**_rounding_fields(places), "rounded_answer": display}


_UNDEFINED_PHRASES = {"kan niet", "bestaat niet", "niet mogelijk", "ongedefinieerd", "niet gedefinieerd"}


def _is_undefined_phrase(answer: str | None) -> bool:
    """True when the answer states that the result does not exist (e.g. 'kan niet')."""
    normalised = " ".join((answer or "").strip().casefold().split()).replace("ë", "e")
    return normalised in _UNDEFINED_PHRASES


def _verify_arithmetic(check_type: str, input_value: str, submitted_answer: str | None,
                       task_instruction: str | None = None) -> dict[str, Any]:
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
    # A euro sign in the answer (€288, € 300,00) is not part of the number.
    answer_text = re.sub(r"^\s*€\s*", "", submitted_answer) if isinstance(submitted_answer, str) else submitted_answer
    try:
        answer = _answer_value(answer_text)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct, rounding_fields = _rounded_compare(value, answer, places, answer_text)
    fields.update(rounding_fields)
    simplified = _is_simplified_root_form(answer_text)
    if simplified is not None:
        fields["is_simplified"] = simplified
    return _result(
        "verified" if correct else "wrong", check_type, input_value, **fields,
        submitted_answer=submitted_answer,
        explanation_nl=("Het antwoord komt overeen met de exacte waarde." if correct
                        else f"Het juiste exacte resultaat is {_serialize_exact(value)}."),
    )


def _verify_answer_equivalence(input_value: str, submitted_answer: str | None,
                               task_instruction: str | None = None) -> dict[str, Any]:
    if submitted_answer is None:
        return _unverifiable("answer_equivalence", input_value, _MathInputError("invalid_answer"))
    ok, reason = _preflight(input_value)
    if not ok:
        return _unverifiable("answer_equivalence", input_value, _MathInputError(reason or "disallowed_characters_or_syntax"))
    try:
        input_value_exact = _numeric_value(input_value)
        answer = _answer_value(submitted_answer)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable("answer_equivalence", input_value, exc, parser_invoked=True)
    correct, rounding_fields = _rounded_compare(input_value_exact, answer, places, submitted_answer)
    simplified = _is_simplified_root_form(submitted_answer)
    if simplified is not None:
        # Equal value and finished form are different questions: √12 for
        # 'vereenvoudig √12' is verified, but is_simplified is false.
        rounding_fields["is_simplified"] = simplified
    return _result(
        "verified" if correct else "wrong", "answer_equivalence", input_value,
        exact_value=_serialize_exact(input_value_exact), submitted_answer=submitted_answer,
        parser_invoked=True, **rounding_fields,
        explanation_nl=("Beide notaties hebben dezelfde exacte waarde." if correct
                        else f"De exacte waarde is {_serialize_exact(input_value_exact)}."),
    )


def _verify_expression_equivalence(input_value: str) -> dict[str, Any]:
    denominators: list[sp.Expr] = []
    root_conditions: list[sp.Expr] = []
    try:
        lhs, rhs = _parse_equation(input_value, denominators, root_conditions)
        difference = sp.simplify(lhs - rhs)
        equivalent = difference == 0
    except _MathInputError as exc:
        return _unverifiable("expression_equivalence", input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    fields: dict[str, Any] = {"equivalent": bool(equivalent), "parser_invoked": True}
    restrictions = _domain_restrictions(denominators) + _root_restrictions(root_conditions)
    if restrictions:
        fields["domain_restrictions"] = restrictions
    return _result(
        "verified" if equivalent else "wrong", "expression_equivalence", input_value,
        **fields,
        explanation_nl=("De twee uitdrukkingen zijn equivalent op het domein waar ze bestaan "
                        "(noemers niet nul, onder een wortel niet negatief)."
                        if equivalent and restrictions else
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
        answers = _parse_answer_list(submitted_answer)
        places = _parse_rounding_places(task_instruction)
        if any(expr.has(unknown) for _, expr in answers):
            raise _MathInputError("invalid_answer")
    except _MathInputError as exc:
        return _unverifiable("equation_solution", input_value, exc, parser_invoked=True)

    # The answer must give the whole solution set: one valid root is not enough
    # when another real root has been left out (x² = 9 needs 3 and -3).
    correct = _matches_solution_set([expr for _, expr in answers], solutions, places)
    misconception = None if correct else _quadratic_misconception(
        lhs - rhs, unknown, [expr for _, expr in answers], solutions, places)
    rounded_answer: str | None = None
    if (places is not None and isinstance(solutions, sp.FiniteSet) and len(solutions) == 1
            and next(iter(solutions)).is_number):
        rounded_answer = _rounded_display(next(iter(solutions)), places)
    mismatched = [name for name, _ in answers if name is not None and name != unknown.name]
    answer_name = mismatched[0] if mismatched else None
    fields: dict[str, Any] = {
        "solution_set": serialized_solutions,
        "submitted_answer": submitted_answer,
        "parser_invoked": True,
        "explanation_nl": ("Het antwoord is correct voor de volledige reële oplossingsverzameling." if correct
                           else "Het gegeven antwoord komt niet overeen met de volledige exacte oplossingsverzameling."),
    }
    if places is not None:
        fields.update(_rounding_fields(places))
        if rounded_answer is not None:
            fields["rounded_answer"] = rounded_answer
    fields["variable"] = unknown.name
    if misconception == "sign_of_b":
        fields["misconception"] = misconception
        fields["explanation_nl"] = ("Alle oplossingen hebben het verkeerde teken: in de abc-formule staat -b. "
                                    "Het juiste antwoord is het tegengestelde van wat er staat.")
    elif misconception == "divided_by_2_not_2a":
        fields["misconception"] = misconception
        fields["explanation_nl"] = ("In de abc-formule wordt gedeeld door 2a, niet door 2. "
                                    "Deel de gegeven oplossingen nog door a.")
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


def _quadratic_misconception(difference: sp.Expr, unknown: sp.Symbol, candidates: Sequence[sp.Expr],
                             solutions: sp.Set, places: int | None) -> str | None:
    """Recognise two abc-formule slips from the whole answer set.

    (b ± √D)/2a gives the negatives of the real roots; (-b ± √D)/2 gives a times
    the real roots. Only for a quadratic with two different real roots.
    """
    if not isinstance(solutions, sp.FiniteSet) or len(solutions) != 2:
        return None
    try:
        poly = sp.Poly(sp.expand(difference), unknown)
    except sp.PolynomialError:
        return None
    if poly.degree() != 2 or any(coefficient.free_symbols for coefficient in poly.all_coeffs()):
        return None
    negated = sp.FiniteSet(*[-root for root in solutions])
    if negated != solutions and _matches_solution_set(candidates, negated, places):
        return "sign_of_b"
    leading = poly.LC()
    if leading not in (1, -1):
        scaled = sp.FiniteSet(*[leading * root for root in solutions])
        if _matches_solution_set(candidates, scaled, places):
            return "divided_by_2_not_2a"
    return None


def _solve_target(numerator: sp.Expr, target: sp.Symbol) -> tuple[sp.Expr, bool]:
    """Solve for the one unknown after filling in values.

    Linear: the single solution. Squared (c² = 41, Pythagoras): two solutions ±v;
    the positive one is taken because the formula describes a length, and the
    second value says so (assumed_positive).
    """
    try:
        poly = sp.Poly(sp.expand(numerator), target)
    except sp.PolynomialError as exc:
        raise _MathInputError("unsupported_solution_set") from exc
    if poly.degree() == 1:
        return _solve_linear(numerator, target), False
    if poly.degree() == 2 and not any(c.free_symbols for c in poly.all_coeffs()):
        solutions = sp.solveset(numerator, target, domain=sp.S.Reals)
        if isinstance(solutions, sp.FiniteSet) and len(solutions) == 2:
            low, high = sorted(solutions, key=lambda value: float(value))
            if sp.simplify(low + high) == 0 and high.is_positive:
                return high, True
    raise _MathInputError("unsupported_solution_set")


def _verify_formula_substitution(input_value: str, substitutions: Any, submitted_answer: str | None,
                                 task_instruction: str | None = None) -> dict[str, Any]:
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
        value, assumed_positive = _solve_target(numerator, target)
        if not value.is_number:
            raise _MathInputError("unsupported_solution_set")
        if value.is_real is False:
            raise _MathInputError("not_real")
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
    if assumed_positive:
        fields["assumed_positive"] = True
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
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct, rounding_fields = _rounded_compare(value, candidate, places, submitted_answer)
    fields.update(rounding_fields)
    fields["submitted_answer"] = submitted_answer
    fields["explanation_nl"] = ("Het antwoord klopt na invullen." if correct
                                else f"Na invullen is {target.name} = {_serialize_exact(value)}.")
    if not correct and assumed_positive and sp.simplify(candidate - value ** 2) == 0:
        # c = 41 instead of c = √41: the root at the end was forgotten.
        fields["misconception"] = "forgot_square_root"
        fields["explanation_nl"] += f" Je hebt {target.name}² berekend; neem nog de wortel."
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
    r"([A-Za-z]+(?:²|³|\^2|\^3|2|3)?(?:/[A-Za-z]+)?)\s*$"
)


def _parse_quantity(value: Any) -> tuple[sp.Expr, str, str]:
    if not isinstance(value, str):
        raise _MathInputError("unit_mismatch")
    match = _QUANTITY_RE.fullmatch(value)
    if not match:
        raise _MathInputError("unit_mismatch")
    number, unit = match.groups()
    # Students and models often type m2 or dm3 instead of m² or dm³.
    unit = unit.replace("^2", "²").replace("^3", "³")
    unit = re.sub(r"2(?=/|$)", "²", unit)
    unit = re.sub(r"3(?=/|$)", "³", unit)
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
        fields.update(_rounding_fields(places))
        fields["rounded_answer"] = _rounded_display(exact_value, places)
    correct = bool(exact_match or rounded_match)
    return _result(
        "verified" if correct else "wrong", check_type, input_value, **fields,
        explanation_nl=("De omrekening klopt exact." if exact_match else
                        "De omrekening klopt na afronden." if correct else
                        f"De hoeveelheid is {_serialize_exact(exact_value)} {answer_unit}."),
    )


def _verify_percentage_change(input_value: str, submitted_answer: str | None,
                              task_instruction: str | None = None) -> dict[str, Any]:
    """Percentage increase or decrease from an old to a new value, e.g. 80 -> 100 = 25%."""
    check_type = "percentage_change"
    parts = re.split(r"\s*(?:->|→)\s*", input_value.strip())
    if len(parts) != 2 or not all(parts):
        return _unverifiable(check_type, input_value, _MathInputError("invalid_percentage_change"))
    try:
        old = _numeric_value(parts[0])
        new = _numeric_value(parts[1])
        if old == 0:
            raise _MathInputError("division_by_zero")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS)
    exact = sp.simplify((new - old) / old * 100)
    fields: dict[str, Any] = {"exact_value": _serialize_exact(exact), "parser_invoked": True}
    if submitted_answer is None:
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=_REASON_NL["no_claim_to_check"])
    try:
        if not isinstance(submitted_answer, str):
            raise _MathInputError("invalid_answer")
        text = submitted_answer.strip()
        lowered = text.casefold()
        direction_word = "afname" if "afname" in lowered else "toename" if "toename" in lowered else None
        text = re.sub(r"\b(toename|afname|van)\b", " ", text, flags=re.IGNORECASE).strip()
        if not text.endswith("%"):
            raise _MathInputError("invalid_percentage_change")
        number_text = text[:-1].strip()
        has_sign = number_text.startswith(("+", "-", "−", "–"))
        value = _numeric_value(number_text)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    if direction_word == "afname":
        value = -abs(value)
    elif direction_word == "toename":
        value = abs(value)
    direction_given = has_sign or direction_word is not None

    def matches(candidate: sp.Expr) -> tuple[bool, dict[str, Any]]:
        # Answer and exact value are both in percent already: no extra scaling.
        return _rounded_compare(exact, candidate, places, "")

    correct, rounding_fields = matches(value)
    fields.update(rounding_fields)
    fields["submitted_answer"] = submitted_answer
    if correct:
        explanation = "De procentuele verandering klopt."
    elif not direction_given and exact < 0 and matches(-value)[0]:
        # 100 -> 80 answered as '20%': the size is right, the direction is missing.
        fields["value_correct"] = True
        fields["direction_missing"] = True
        size = _serialize_exact(abs(exact))
        explanation = f"De grootte klopt, maar het is een afname: schrijf -{size}% of {size}% afname."
    else:
        explanation = f"De procentuele verandering is {_serialize_exact(exact)}%."
        if new != 0:
            wrong_base = sp.simplify((new - old) / new * 100)
            if _matches_value(wrong_base, value, places, direction_given):
                fields["misconception"] = "divided_by_new_value"
                explanation += " Deel het verschil door de oude waarde, niet door de nieuwe."
    fields["explanation_nl"] = explanation
    return _result("verified" if correct else "wrong", check_type, input_value, **fields)


def _matches_value(target: sp.Expr, value: sp.Expr, places: int | None, direction_given: bool) -> bool:
    """True when value equals target (or, without a direction, its size)."""
    candidates = [value] if direction_given else [value, -value]
    for candidate in candidates:
        if _rounded_compare(target, candidate, places, "")[0]:
            return True
    return False


# --- Inequalities -----------------------------------------------------------
# A solution set of a linear inequality is one interval. It is kept as a small
# tuple (lo, lo_closed, hi, hi_closed); None for lo or hi means no bound on
# that side. _EMPTY is the empty set. SymPy's own inequality solver is not
# used: a linear inequality only needs the sign of one coefficient, and this
# keeps every step easy to check by hand.
_EMPTY = "empty"
_FLIP = {"<": ">", ">": "<", "≤": "≥", "≥": "≤"}
_INEQ_SPLIT_RE = re.compile(r"(≤|≥|<|>)")
_INTERVAL_MARKERS = set("⟨⟩[]∈←→∞")


def _normalize_inequality_signs(text: str) -> str:
    return text.replace("<=", "≤").replace(">=", "≥").replace("=<", "≤").replace("=>", "≥")


def _split_inequality(text: str) -> tuple[list[str], list[str]]:
    """'-4 < 2x + 2 ≤ 8' -> (['-4', '2x + 2', '8'], ['<', '≤']); one or two signs only."""
    pieces = _INEQ_SPLIT_RE.split(_normalize_inequality_signs(text))
    parts = [piece.strip() for piece in pieces[0::2]]
    ops = pieces[1::2]
    if not ops or len(ops) > 2 or any(not part for part in parts) or any("=" in part for part in parts):
        raise _MathInputError("invalid_inequality")
    if len(ops) == 2:
        lower_type = {"<", "≤"}
        if (ops[0] in lower_type) != (ops[1] in lower_type):
            # 1 < x > 3 is not a chain a reader can trust.
            raise _MathInputError("invalid_inequality")
    return parts, ops


def _sign_of(value: sp.Expr) -> int:
    value = sp.simplify(value)
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    raise _MathInputError("unsupported_solution_set")


def _holds(op: str, value: sp.Expr) -> bool:
    """True when 'value op 0' holds for a number."""
    sign = _sign_of(value)
    return {"<": sign < 0, "≤": sign <= 0, ">": sign > 0, "≥": sign >= 0}[op]


def _ray(op: str, bound: sp.Expr) -> tuple:
    """The set 'x op bound'."""
    if op == "<":
        return (None, False, bound, False)
    if op == "≤":
        return (None, False, bound, True)
    if op == ">":
        return (bound, False, None, False)
    return (bound, True, None, False)


def _solve_linear_inequality(left: sp.Expr, right: sp.Expr, op: str, unknown: sp.Symbol):
    difference = sp.expand(left - right)
    try:
        poly = sp.Poly(difference, unknown)
    except sp.PolynomialError as exc:
        raise _MathInputError("unsupported_solution_set") from exc
    if poly.degree() > 1:
        # Quadratic inequalities need a sign chart; not built yet.
        raise _MathInputError("unsupported_solution_set")
    coefficient = sp.simplify(poly.nth(1))
    constant = sp.simplify(poly.nth(0))
    if coefficient.free_symbols or constant.free_symbols:
        raise _MathInputError("unsupported_solution_set")
    sign = _sign_of(coefficient)
    if sign == 0:
        return (None, False, None, False) if _holds(op, constant) else _EMPTY
    bound = sp.simplify(-constant / coefficient)
    # Dividing by a negative coefficient flips the sign.
    return _ray(op if sign > 0 else _FLIP[op], bound)


def _intersect(first, second):
    if first == _EMPTY or second == _EMPTY:
        return _EMPTY
    lo, lo_closed = first[0], first[1]
    if second[0] is not None:
        if lo is None:
            lo, lo_closed = second[0], second[1]
        else:
            sign = _sign_of(second[0] - lo)
            if sign > 0:
                lo, lo_closed = second[0], second[1]
            elif sign == 0:
                lo_closed = lo_closed and second[1]
    hi, hi_closed = first[2], first[3]
    if second[2] is not None:
        if hi is None:
            hi, hi_closed = second[2], second[3]
        else:
            sign = _sign_of(second[2] - hi)
            if sign < 0:
                hi, hi_closed = second[2], second[3]
            elif sign == 0:
                hi_closed = hi_closed and second[3]
    if lo is not None and hi is not None:
        width = _sign_of(hi - lo)
        if width < 0 or (width == 0 and not (lo_closed and hi_closed)):
            return _EMPTY
    return (lo, lo_closed, hi, hi_closed)


def _interval_text(interval, name: str) -> str:
    if interval == _EMPTY:
        return "empty"
    lo, lo_closed, hi, hi_closed = interval
    if lo is None and hi is None:
        return "all_reals"
    if lo is None:
        return f"{name} {'≤' if hi_closed else '<'} {_serialize_exact(hi)}"
    if hi is None:
        return f"{name} {'≥' if lo_closed else '>'} {_serialize_exact(lo)}"
    if _sign_of(hi - lo) == 0:
        return f"{name} = {_serialize_exact(lo)}"
    return (f"{_serialize_exact(lo)} {'≤' if lo_closed else '<'} {name} "
            f"{'≤' if hi_closed else '<'} {_serialize_exact(hi)}")


def _interval_nl(interval, name: str) -> str:
    text = _interval_text(interval, name)
    if text == "empty":
        return "er is geen oplossing"
    if text == "all_reals":
        return "alle reële getallen voldoen"
    return text


def _solve_inequality_input(input_value: str, variable: str | None):
    if not isinstance(input_value, str) or not input_value.strip():
        raise _MathInputError("invalid_inequality")
    if len(input_value) > MAX_INPUT_LENGTH:
        raise _MathInputError("input_too_long")
    parts, ops = _split_inequality(input_value)
    expressions: list[sp.Expr] = []
    for part in parts:
        ok, reason = _preflight(part)
        if not ok:
            raise _MathInputError(reason or "disallowed_characters_or_syntax")
        denominators: list[sp.Expr] = []
        root_conditions: list[sp.Expr] = []
        expressions.append(_parse_expr(part, denominators, root_conditions))
        if denominators:
            # A letter in a denominator can change sign: not a linear inequality.
            raise _MathInputError("variable_denominator_unsupported")
        if root_conditions:
            raise _MathInputError("unsupported_solution_set")
    symbols: set = set()
    for expression in expressions:
        symbols |= expression.free_symbols
    unknown = _pick_unknown(symbols, variable)
    if symbols - {unknown}:
        # Parameters (for which p ...) are planned, not supported.
        raise _MathInputError("unsupported_solution_set")
    interval = _solve_linear_inequality(expressions[0], expressions[1], ops[0], unknown)
    if len(ops) == 2:
        interval = _intersect(interval, _solve_linear_inequality(expressions[1], expressions[2], ops[1], unknown))
    return unknown, interval


def _parse_inequality_answer(text: str) -> tuple[str, tuple]:
    """Read 'x < 3', '3 > x', 'x >= 2' or '-3 < x ≤ 3'. Returns the letter and the interval."""
    parts, ops = _split_inequality(text)
    letters = [bool(_SINGLE_LETTER_RE.fullmatch(part)) for part in parts]
    if len(ops) == 1:
        if letters[0] and not letters[1]:
            return parts[0], _ray(ops[0], _numeric_value(parts[1]))
        if letters[1] and not letters[0]:
            return parts[1], _ray(_FLIP[ops[0]], _numeric_value(parts[0]))
        raise _MathInputError("invalid_answer")
    if not letters[1] or letters[0] or letters[2]:
        raise _MathInputError("invalid_answer")
    first, last = _numeric_value(parts[0]), _numeric_value(parts[2])
    if ops[0] in ("<", "≤"):
        # a < x ≤ b
        return parts[1], (first, ops[0] == "≤", last, ops[1] == "≤")
    # a > x ≥ b is the same as b ≤ x < a
    return parts[1], (last, ops[1] == "≥", first, ops[0] == "≥")


def _same_bound(exact: sp.Expr | None, answer: sp.Expr | None, places: int | None) -> bool:
    if exact is None or answer is None:
        return exact is None and answer is None
    return _rounded_compare(exact, answer, places, "")[0]


def _same_interval(exact, answer, places: int | None) -> bool:
    if exact == _EMPTY or answer == _EMPTY:
        return exact == answer
    return (_same_bound(exact[0], answer[0], places) and _same_bound(exact[2], answer[2], places)
            and (exact[0] is None or exact[1] == answer[1])
            and (exact[2] is None or exact[3] == answer[3]))


def _inequality_misconception(exact, answer, places: int | None) -> str | None:
    if exact == _EMPTY or answer == _EMPTY:
        return None
    one_sided_exact = (exact[0] is None) != (exact[2] is None)
    one_sided_answer = (answer[0] is None) != (answer[2] is None)
    if one_sided_exact and one_sided_answer and (exact[0] is None) != (answer[0] is None):
        exact_bound = exact[0] if exact[0] is not None else exact[2]
        answer_bound = answer[0] if answer[0] is not None else answer[2]
        if _same_bound(exact_bound, answer_bound, places):
            return "direction_reversed"
    if (_same_bound(exact[0], answer[0], places) and _same_bound(exact[2], answer[2], places)
            and not _same_interval(exact, answer, places)):
        return "wrong_bound_type"
    return None


_NO_SOLUTION_PHRASES = {"geen oplossing", "geen oplossingen"}
_ALL_REALS_PHRASES = {"alle getallen", "alle reele getallen", "alle reele waarden", "alle x"}


def _verify_inequality_solution(input_value: str, submitted_answer: str | None,
                                task_instruction: str | None = None,
                                variable: str | None = None) -> dict[str, Any]:
    check_type = "inequality_solution"
    try:
        unknown, exact = _solve_inequality_input(input_value, variable)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS | {"invalid_inequality"})
    name = unknown.name
    solution_text = _interval_text(exact, name)
    base = {"solution_set": solution_text, "variable": name, "parser_invoked": True}

    if submitted_answer is None or not str(submitted_answer).strip():
        error = _MathInputError("no_claim_to_check")
        return _result("unverifiable", check_type, input_value, reason=error.reason,
                       explanation_nl=f"{error.message_nl} De oplossing is: {_interval_nl(exact, name)}.", **base)
    base["submitted_answer"] = submitted_answer
    if any(char in _INTERVAL_MARKERS for char in submitted_answer):
        error = _MathInputError("interval_notation_not_supported")
        return _result("unverifiable", check_type, input_value, reason=error.reason,
                       explanation_nl=error.message_nl, **base)

    phrase = " ".join(submitted_answer.strip().casefold().split()).replace("ë", "e")
    if phrase in _NO_SOLUTION_PHRASES or phrase in _ALL_REALS_PHRASES:
        answer = _EMPTY if phrase in _NO_SOLUTION_PHRASES else (None, False, None, False)
        answer_name = name
    else:
        try:
            answer_name, answer = _parse_inequality_answer(submitted_answer)
        except _MathInputError as exc:
            return _result("unverifiable", check_type, input_value, reason=exc.reason,
                           explanation_nl=exc.message_nl, **base)

    correct = _same_interval(exact, answer, places)
    fields = dict(base)
    if places is not None:
        fields.update(_rounding_fields(places))
    if answer_name != name:
        fields["variable_mismatch"] = True
        fields["value_correct"] = correct
        fields["explanation_nl"] = (f"De ongelijkheid wordt opgelost naar {name}, maar het antwoord gebruikt "
                                    f"{answer_name}. De oplossing is: {_interval_nl(exact, name)}.")
        return _result("wrong", check_type, input_value, **fields)
    if correct:
        fields["explanation_nl"] = f"Het antwoord klopt: {_interval_nl(exact, name)}."
        return _result("verified", check_type, input_value, **fields)

    misconception = _inequality_misconception(exact, answer, places)
    if misconception == "direction_reversed":
        explanation = ("De grens klopt, maar het ongelijkheidsteken wijst de verkeerde kant op. "
                       "Let op: bij delen of vermenigvuldigen met een negatief getal klapt het teken om.")
    elif misconception == "wrong_bound_type":
        explanation = "De grenzen kloppen, maar < en ≤ (of > en ≥) zijn verwisseld."
    else:
        explanation = "Het antwoord komt niet overeen met de oplossing."
    fields["explanation_nl"] = f"{explanation} De oplossing is: {_interval_nl(exact, name)}."
    if misconception is not None:
        fields["misconception"] = misconception
    return _result("wrong", check_type, input_value, **fields)


# --- Systems of two linear equations ---------------------------------------
_SYSTEM_SPLIT_RE = re.compile(r"\s*;\s*|\s*\n\s*")
_SYSTEM_ANSWER_SPLIT_RE = re.compile(r"\s*;\s*|\s+en\s+|\s*\n\s*|\s*,\s*(?=[A-Za-z]\s*=)", re.IGNORECASE)
_SYSTEM_ASSIGNMENT_RE = re.compile(r"\s*([A-Za-z])\s*=\s*(.+?)\s*")
_INFINITE_PHRASES = {"oneindig veel oplossingen", "oneindig veel", "oneindig veel oplossing"}


def _linear_coefficients(difference: sp.Expr, unknowns: Sequence[sp.Symbol]) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """a, b, c in a·u + b·v + c = 0; fails closed for anything that is not linear."""
    try:
        poly = sp.Poly(sp.expand(difference), *unknowns)
    except sp.PolynomialError as exc:
        raise _MathInputError("unsupported_solution_set") from exc
    if poly.total_degree() > 1:
        raise _MathInputError("unsupported_solution_set")
    first, second = unknowns
    a = poly.coeff_monomial(first)
    b = poly.coeff_monomial(second)
    c = poly.coeff_monomial(1)
    for value in (a, b, c):
        if value.free_symbols:
            raise _MathInputError("unsupported_solution_set")
    return sp.nsimplify(a), sp.nsimplify(b), sp.nsimplify(c)


def _solve_system_input(input_value: Any):
    """Returns (unknowns, differences, kind, values) with kind in unique / empty / infinite."""
    if not isinstance(input_value, str) or not input_value.strip():
        raise _MathInputError("invalid_system")
    if len(input_value) > MAX_INPUT_LENGTH:
        raise _MathInputError("input_too_long")
    equations = [part for part in _SYSTEM_SPLIT_RE.split(input_value.strip()) if part.strip()]
    if len(equations) != 2:
        raise _MathInputError("invalid_system")
    differences: list[sp.Expr] = []
    for equation in equations:
        denominators: list[sp.Expr] = []
        root_conditions: list[sp.Expr] = []
        lhs, rhs = _parse_equation(equation, denominators, root_conditions)
        if denominators:
            raise _MathInputError("variable_denominator_unsupported")
        if root_conditions:
            raise _MathInputError("unsupported_solution_set")
        differences.append(sp.expand(lhs - rhs))
    symbols: set = set()
    for difference in differences:
        symbols |= difference.free_symbols
    if len(symbols) != 2:
        # Three letters, or only one: not a system of two equations in two unknowns.
        raise _MathInputError("unsupported_solution_set")
    unknowns = sorted(symbols, key=lambda symbol: symbol.name)
    (a1, b1, c1), (a2, b2, c2) = (_linear_coefficients(d, unknowns) for d in differences)
    determinant = sp.simplify(a1 * b2 - a2 * b1)
    if determinant != 0:
        # Cramer's rule for a1·u + b1·v = -c1 and a2·u + b2·v = -c2.
        first = sp.simplify((b1 * c2 - b2 * c1) / determinant)
        second = sp.simplify((c1 * a2 - a1 * c2) / determinant)
        return unknowns, differences, "unique", {unknowns[0].name: first, unknowns[1].name: second}
    matrix = sp.Matrix([[a1, b1], [a2, b2]])
    augmented = sp.Matrix([[a1, b1, c1], [a2, b2, c2]])
    if matrix.rank() != augmented.rank():
        return unknowns, differences, "empty", {}
    return unknowns, differences, "infinite", {}


def _serialize_system(kind: str, values: dict[str, sp.Expr]) -> dict[str, Any]:
    if kind == "unique":
        return {"kind": "unique", "values": {name: _serialize_exact(value) for name, value in values.items()}}
    return {"kind": kind}


def _system_nl(kind: str, values: dict[str, sp.Expr]) -> str:
    if kind == "empty":
        return "het stelsel heeft geen oplossing"
    if kind == "infinite":
        return "het stelsel heeft oneindig veel oplossingen"
    return " en ".join(f"{name} = {_serialize_exact(value)}" for name, value in values.items())


def _parse_system_answer(text: str) -> dict[str, sp.Expr]:
    """Read 'x = 3, y = 1', 'x = 3 en y = 1' or 'x = 3; y = 1' (decimal commas allowed)."""
    answers: dict[str, sp.Expr] = {}
    for part in _SYSTEM_ANSWER_SPLIT_RE.split(text.strip()):
        if not part.strip():
            raise _MathInputError("invalid_answer")
        match = _SYSTEM_ASSIGNMENT_RE.fullmatch(part)
        if not match:
            raise _MathInputError("invalid_answer")
        name, value_text = match.groups()
        if name in answers:
            raise _MathInputError("invalid_answer")
        try:
            answers[name] = _numeric_value(value_text)
        except _MathInputError as exc:
            if exc.reason in _PREFLIGHT_REASONS:
                raise
            raise _MathInputError("invalid_answer") from exc
    return answers


def _verify_system_solution(input_value: str, submitted_answer: str | None,
                            task_instruction: str | None = None) -> dict[str, Any]:
    check_type = "system_solution"
    try:
        unknowns, differences, kind, values = _solve_system_input(input_value)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS | {"invalid_system"})
    solution_text = _system_nl(kind, values)
    base: dict[str, Any] = {"solution_set": _serialize_system(kind, values), "parser_invoked": True}

    if submitted_answer is None or not str(submitted_answer).strip():
        error = _MathInputError("no_claim_to_check")
        return _result("unverifiable", check_type, input_value, reason=error.reason,
                       explanation_nl=f"{error.message_nl} De oplossing: {solution_text}.", **base)
    base["submitted_answer"] = submitted_answer

    phrase = " ".join(submitted_answer.strip().casefold().split())
    if phrase in {"geen oplossing", "geen oplossingen"} or phrase in _INFINITE_PHRASES:
        claimed = "empty" if phrase.startswith("geen") else "infinite"
        correct = claimed == kind
        return _result("verified" if correct else "wrong", check_type, input_value, **base,
                       explanation_nl=(f"Het antwoord klopt: {solution_text}." if correct
                                       else f"Het antwoord klopt niet: {solution_text}."))

    try:
        answers = _parse_system_answer(submitted_answer)
    except _MathInputError as exc:
        return _result("unverifiable", check_type, input_value, reason=exc.reason,
                       explanation_nl=("Schrijf het antwoord met de letters erbij, bijvoorbeeld x = 3 en y = 1."
                                       if exc.reason == "invalid_answer" else exc.message_nl), **base)

    names = [symbol.name for symbol in unknowns]
    fields = dict(base)
    if places is not None:
        fields.update(_rounding_fields(places))
    if kind != "unique":
        fields["explanation_nl"] = f"Het antwoord geeft één oplossing, maar {solution_text}."
        return _result("wrong", check_type, input_value, **fields)

    unknown_letters = [name for name in answers if name not in names]
    if unknown_letters:
        fields["variable_mismatch"] = True
        fields["explanation_nl"] = (f"Het stelsel gaat over {names[0]} en {names[1]}, maar het antwoord gebruikt "
                                    f"{', '.join(unknown_letters)}. De oplossing: {solution_text}.")
        return _result("wrong", check_type, input_value, **fields)
    missing = [name for name in names if name not in answers]
    if missing:
        fields["misconception"] = "missing_variable"
        fields["explanation_nl"] = (f"Het antwoord mist {', '.join(missing)}; een stelsel heeft een waarde "
                                    f"voor beide letters nodig. De oplossing: {solution_text}.")
        return _result("wrong", check_type, input_value, **fields)

    def same(exact: sp.Expr, given: sp.Expr) -> bool:
        return _rounded_compare(exact, given, places, "")[0]

    if all(same(values[name], answers[name]) for name in names):
        fields["explanation_nl"] = f"Het antwoord klopt: {solution_text}."
        return _result("verified", check_type, input_value, **fields)

    substitution = {symbol: answers[symbol.name] for symbol in unknowns}
    fields["satisfied_equations"] = [index + 1 for index, difference in enumerate(differences)
                                     if sp.simplify(difference.subs(substitution)) == 0]
    if same(values[names[0]], answers[names[1]]) and same(values[names[1]], answers[names[0]]):
        fields["misconception"] = "values_swapped"
        explanation = f"De getallen kloppen, maar {names[0]} en {names[1]} zijn verwisseld."
    elif fields["satisfied_equations"]:
        explanation = (f"Het antwoord past alleen in vergelijking {fields['satisfied_equations'][0]}; "
                       "controleer altijd beide vergelijkingen.")
    else:
        explanation = "Het antwoord past in geen van beide vergelijkingen."
    fields["explanation_nl"] = f"{explanation} De oplossing: {solution_text}."
    return _result("wrong", check_type, input_value, **fields)


# --- Function notation f(x) -------------------------------------------------
# Only this check type reads f(...) as a function call. A call is replaced by
# the definition with the argument in brackets, as a student would write it
# out: f(x) = 2x² - 3x + 1 makes f(-2) the text (2(-2)² - 3(-2) + 1). The
# normal safe parser then does the rest, including division by zero.
_FUNC_DEF_RE = re.compile(r"\s*([A-Za-z])\s*\(\s*([A-Za-z])\s*\)\s*=\s*(.+?)\s*")
_FUNC_CALL_ANSWER_RE = re.compile(r"\s*([A-Za-z])\s*\((.+)\)\s*=\s*(.+?)\s*")


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise _MathInputError("invalid_function")


def _expand_calls(text: str, definitions: dict[str, tuple[str, str]], depth: int = 0) -> str:
    """Replace every call f(...) of a defined function by its written-out definition."""
    if depth > 3:
        raise _MathInputError("invalid_function")
    variables = {variable for variable, _ in definitions.values()}
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        is_call = (char.isascii() and char.isalpha() and index + 1 < len(text) and text[index + 1] == "(")
        if is_call and char in definitions:
            close = _matching_paren(text, index + 1)
            argument = text[index + 2:close]
            if not argument.strip():
                raise _MathInputError("invalid_function")
            argument = _expand_calls(argument, definitions, depth + 1)
            variable, body = definitions[char]
            out.append("(" + body.replace(variable, f"({argument})") + ")")
            index = close + 1
            continue
        if is_call and char not in variables:
            previous = text[index - 1] if index > 0 else " "
            if not (previous.isalnum() or previous == ")"):
                # g(2) while only f is defined: a call of an unknown function.
                raise _MathInputError("invalid_function")
        out.append(char)
        index += 1
    return "".join(out)


def _parse_function_input(input_value: Any) -> tuple[dict[str, tuple[str, str]], str]:
    if not isinstance(input_value, str) or not input_value.strip():
        raise _MathInputError("invalid_function")
    if len(input_value) > MAX_INPUT_LENGTH:
        raise _MathInputError("input_too_long")
    parts = [part for part in _SYSTEM_SPLIT_RE.split(input_value.strip()) if part.strip()]
    if len(parts) < 2:
        raise _MathInputError("invalid_function")
    definitions: dict[str, tuple[str, str]] = {}
    for part in parts[:-1]:
        match = _FUNC_DEF_RE.fullmatch(part)
        if not match:
            raise _MathInputError("invalid_function")
        name, variable, body = match.groups()
        if name in definitions or name == variable:
            raise _MathInputError("invalid_function")
        body = _normalize_percent_phrases(body)
        ok, reason = _preflight(body)
        if not ok:
            raise _MathInputError(reason or "disallowed_characters_or_syntax")
        definitions[name] = (variable, body)
    return definitions, parts[-1].strip()


def _single_call(task: str) -> tuple[str, str] | None:
    """('f', '-2') when the task is exactly one call such as f(-2)."""
    match = re.fullmatch(r"\s*([A-Za-z])\((.*)\)\s*", task)
    if not match:
        return None
    open_index = task.index("(")
    if _matching_paren(task, open_index) != len(task.rstrip()) - 1:
        return None
    return match.group(1), match.group(2)


def _verify_function_notation(input_value: str, submitted_answer: str | None,
                              task_instruction: str | None = None,
                              variable: str | None = None) -> dict[str, Any]:
    check_type = "function_notation"
    try:
        definitions, task = _parse_function_input(input_value)
        expanded = _expand_calls(task, definitions)
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc,
                             parser_invoked=exc.reason not in _PREFLIGHT_REASONS | {"invalid_function"})

    if "=" in task:
        # An equation such as f(x) = 8 or f(x) = g(x): solved as equation_solution.
        first_variable = next(iter(definitions.values()))[0]
        result = _verify_equation_solution(expanded, submitted_answer, task_instruction,
                                           variable or first_variable)
        result["check_type"] = check_type
        result["input"] = input_value
        result["expanded_equation"] = expanded
        return result

    call = _single_call(task)
    ok, reason = _preflight(expanded)
    if not ok:
        return _unverifiable(check_type, input_value, _MathInputError(reason or "disallowed_characters_or_syntax"))
    try:
        value = sp.simplify(_parse_expr(expanded, []))
    except _MathInputError as exc:
        if exc.reason == "division_by_zero":
            if submitted_answer is None or not str(submitted_answer).strip():
                return _result("unverifiable", check_type, input_value, reason="no_claim_to_check",
                               explanation_nl="De functie bestaat niet voor dit getal (deling door nul); er is geen antwoord om te controleren.",
                               parser_invoked=True)
            undefined = _is_undefined_phrase(submitted_answer)
            return _result("verified" if undefined else "wrong", check_type, input_value,
                           reason=exc.reason, submitted_answer=submitted_answer, parser_invoked=True,
                           explanation_nl=("De functiewaarde bestaat niet: de noemer wordt nul. Het antwoord zegt dit correct."
                                           if undefined else "De functiewaarde bestaat niet: de noemer wordt nul."))
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)

    fields: dict[str, Any] = {"exact_value": _serialize_exact(value), "parser_invoked": True}
    if submitted_answer is None or not str(submitted_answer).strip():
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"{_REASON_NL['no_claim_to_check']} De waarde is {_serialize_exact(value)}.")
    fields["submitted_answer"] = submitted_answer

    answer_text = submitted_answer
    argument_mismatch = False
    answer_call = _FUNC_CALL_ANSWER_RE.fullmatch(submitted_answer)
    if answer_call:
        name, argument_text, answer_text = answer_call.groups()
        if call is None:
            argument_mismatch = True
        else:
            try:
                same_argument = (name == call[0] and sp.simplify(
                    _safe_parse_expr(argument_text, allow_variable_denominators=True)
                    - _safe_parse_expr(call[1], allow_variable_denominators=True)) == 0)
            except _MathInputError:
                same_argument = False
            argument_mismatch = not same_argument

    try:
        if value.free_symbols:
            answer = _safe_parse_expr(answer_text, allow_variable_denominators=True)
            correct = sp.simplify(value - answer) == 0
            rounding_fields: dict[str, Any] = {}
        else:
            answer = _numeric_value(answer_text)
            correct, rounding_fields = _rounded_compare(value, answer, places, answer_text)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    fields.update(rounding_fields)

    if argument_mismatch:
        fields["argument_mismatch"] = True
        fields["value_correct"] = correct
        fields["explanation_nl"] = (f"Gevraagd is {task}; het antwoord noemt een ander getal tussen de haakjes. "
                                    f"De juiste waarde is {_serialize_exact(value)}.")
        return _result("wrong", check_type, input_value, **fields)
    if correct:
        fields["explanation_nl"] = f"Het antwoord klopt: {task} = {_serialize_exact(value)}."
        return _result("verified", check_type, input_value, **fields)

    if call is not None and not value.free_symbols:
        name, argument_text = call
        try:
            argument = _numeric_value(argument_text)
        except _MathInputError:
            argument = None
        if argument is not None and argument.is_negative:
            variable_name, body = definitions[name]
            unknown = _symbol(variable_name)
            body_expr = _parse_expr(body, [])
            size = abs(argument)
            # What a calculator gives for -2² without brackets: -(2²).
            careless = body_expr.replace(
                lambda e: e.is_Pow and e.base == unknown and e.exp.is_Integer and e.exp.is_even and e.exp > 0,
                lambda e: -(size ** e.exp))
            careless = sp.simplify(careless.subs(unknown, argument))
            if not careless.free_symbols and _rounded_compare(careless, answer, places, answer_text)[0]:
                fields["misconception"] = "negative_base_not_in_brackets"
                fields["explanation_nl"] = (f"Het lijkt erop dat ({_serialize_exact(argument)})² als "
                                            f"-{_serialize_exact(size)}² is uitgerekend. Zet een negatief getal "
                                            f"tussen haakjes. De juiste waarde is {_serialize_exact(value)}.")
                return _result("wrong", check_type, input_value, **fields)
    fields["explanation_nl"] = f"Het antwoord klopt niet: {task} = {_serialize_exact(value)}."
    return _result("wrong", check_type, input_value, **fields)


# --- Statistics: gemiddelde, mediaan, modus, spreidingsbreedte --------------
_STAT_MEASURES = {
    "gemiddelde": "mean",
    "mediaan": "median",
    "modus": "mode",
    "spreidingsbreedte": "range",
    "variatiebreedte": "range",
    "bereik": "range",
}
_STAT_NAMES_NL = {"mean": "het gemiddelde", "median": "de mediaan", "mode": "de modus", "range": "de spreidingsbreedte"}
_STAT_INPUT_RE = re.compile(r"\s*([A-Za-zÀ-ÿ]+)\s*:\s*(.+?)\s*", re.DOTALL)
_STAT_VALUE_RE = re.compile(r"[-−–]?\d+(?:,\d+)?")
MAX_STAT_VALUES = 100


def _parse_stat_values(data: str) -> list[sp.Rational]:
    """'4, 7, 7' or '1,5; 2,5': ';' or a comma plus space separates; a comma between digits is decimal."""
    items = data.split(";") if ";" in data else re.split(r",\s+", data)
    values: list[sp.Rational] = []
    for item in items:
        item = item.strip()
        if not item:
            raise _MathInputError("invalid_statistics_input")
        if "." in item:
            raise _MathInputError("ambiguous_dot_notation")
        if not _STAT_VALUE_RE.fullmatch(item):
            if "," in item:
                raise _MathInputError("ambiguous_comma_notation")
            raise _MathInputError("invalid_statistics_input")
        values.append(sp.Rational(item.replace("−", "-").replace("–", "-").replace(",", ".")))
    if not values or len(values) > MAX_STAT_VALUES:
        raise _MathInputError("invalid_statistics_input")
    return values


def _median(values: Sequence[sp.Expr]) -> sp.Expr:
    count = len(values)
    middle = count // 2
    if count % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _modes(values: Sequence[sp.Expr]) -> list[sp.Expr]:
    counts: dict[sp.Expr, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) > 1 and len(set(counts.values())) == 1:
        return []  # every value occurs equally often: geen modus
    top = max(counts.values())
    return sorted((value for value, count in counts.items() if count == top), key=lambda v: float(v))


def _modes_text(modes: Sequence[sp.Expr]) -> str:
    if not modes:
        return "geen modus"
    return " en ".join(_serialize_exact(mode) for mode in modes)


def _verify_statistics(input_value: str, submitted_answer: str | None,
                       task_instruction: str | None = None) -> dict[str, Any]:
    check_type = "statistics"
    try:
        if not isinstance(input_value, str) or not input_value.strip():
            raise _MathInputError("invalid_statistics_input")
        if len(input_value) > MAX_INPUT_LENGTH:
            raise _MathInputError("input_too_long")
        match = _STAT_INPUT_RE.fullmatch(input_value)
        if not match:
            raise _MathInputError("invalid_statistics_input")
        measure = _STAT_MEASURES.get(match.group(1).casefold())
        if measure is None:
            raise _MathInputError("unsupported_statistic")
        raw = _parse_stat_values(match.group(2))
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc)

    ordered = sorted(raw, key=lambda v: float(v))
    mean = sum(raw, sp.Integer(0)) / len(raw)
    median = _median(ordered)
    modes = _modes(ordered)
    fields: dict[str, Any] = {
        "measure": measure,
        "count": len(raw),
        "sorted_values": [_serialize_exact(value) for value in ordered],
        "parser_invoked": True,
    }
    if measure == "mode":
        fields["exact_value"] = _modes_text(modes)
        fields["modes"] = [_serialize_exact(mode) for mode in modes]
    else:
        exact = {"mean": mean, "median": median, "range": ordered[-1] - ordered[0]}[measure]
        fields["exact_value"] = _serialize_exact(exact)
    name_nl = _STAT_NAMES_NL[measure]

    if submitted_answer is None or not str(submitted_answer).strip():
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"{_REASON_NL['no_claim_to_check']} {name_nl.capitalize()} is {fields['exact_value']}.")
    fields["submitted_answer"] = submitted_answer

    if measure == "mode":
        phrase = " ".join(submitted_answer.strip().casefold().split())
        try:
            if phrase in {"geen modus", "geen", "er is geen modus"}:
                given: list[sp.Expr] = []
            else:
                given = [_numeric_value(part) for part in re.split(r"\s+en\s+|\s*;\s*|,\s+", submitted_answer.strip())
                         if part.strip()]
                if not given:
                    raise _MathInputError("invalid_answer")
        except _MathInputError as exc:
            return _unverifiable(check_type, input_value, exc, parser_invoked=True)
        correct = sorted(given, key=lambda v: float(v)) == list(modes)
        if correct:
            fields["explanation_nl"] = f"Het antwoord klopt: {name_nl} is {fields['exact_value']}."
        elif len(modes) > 1 and set(given) < set(modes):
            fields["explanation_nl"] = (f"Er zijn meerdere waarden die het vaakst voorkomen: {fields['exact_value']}. "
                                        "Noem ze allemaal.")
        else:
            fields["explanation_nl"] = f"Het antwoord klopt niet: {name_nl} is {fields['exact_value']}."
        return _result("verified" if correct else "wrong", check_type, input_value, **fields)

    try:
        answer = _numeric_value(submitted_answer)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct, rounding_fields = _rounded_compare(exact, answer, places, submitted_answer)
    fields.update(rounding_fields)
    if correct:
        fields["explanation_nl"] = f"Het antwoord klopt: {name_nl} is {fields['exact_value']}."
        return _result("verified", check_type, input_value, **fields)

    def same(value: sp.Expr) -> bool:
        return _rounded_compare(value, answer, places, submitted_answer)[0]

    misconception = None
    explanation = f"Het antwoord klopt niet: {name_nl} is {fields['exact_value']}."
    if measure == "median":
        unsorted_middle = _median(raw)
        if unsorted_middle != median and same(unsorted_middle):
            misconception = "median_not_sorted"
            explanation = ("Zet de getallen eerst op volgorde; daarna is de mediaan het middelste getal. "
                           f"De mediaan is {fields['exact_value']}.")
        elif len(raw) % 2 == 0 and (same(ordered[len(raw) // 2 - 1]) or same(ordered[len(raw) // 2])):
            misconception = "median_even_count"
            explanation = ("Bij een even aantal getallen is de mediaan het gemiddelde van de twee middelste getallen. "
                           f"De mediaan is {fields['exact_value']}.")
        elif mean != median and same(mean):
            misconception = "confused_with_mean"
            explanation = f"Dit is het gemiddelde, niet de mediaan. De mediaan is {fields['exact_value']}."
    elif measure == "mean" and mean != median and same(median):
        misconception = "confused_with_median"
        explanation = f"Dit is de mediaan, niet het gemiddelde. Het gemiddelde is {fields['exact_value']}."
    if misconception:
        fields["misconception"] = misconception
    fields["explanation_nl"] = explanation
    return _result("wrong", check_type, input_value, **fields)


# --- Scientific notation ------------------------------------------------------
_SCI_FORM_RE = re.compile(r"\s*(-?\d+(?:,\d+)?)\s*[·×*]\s*10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?\s*")
_CALCULATOR_FORM_RE = re.compile(r"\s*(-?\d+(?:,\d+)?)\s*[Ee]\s*([+-]?\d+)\s*")
_SUPERSCRIPT_OUT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _decimal_of(value: sp.Expr) -> Decimal | None:
    """The exact decimal of a rational with a terminating expansion, else None."""
    if not value.is_Rational:
        return None
    q = int(value.q)
    while q % 2 == 0:
        q //= 2
    while q % 5 == 0:
        q //= 5
    if q != 1:
        return None
    with localcontext() as ctx:
        ctx.prec = 80
        return Decimal(int(value.p)) / Decimal(int(value.q))


def _power_of_ten(value: sp.Expr) -> int:
    """n with 10^n ≤ |value| < 10^(n+1)."""
    with localcontext() as ctx:
        ctx.prec = 60
        if value.is_Rational:
            decimal_value = Decimal(int(value.p)) / Decimal(int(value.q))
        else:
            decimal_value = Decimal(str(sp.N(value, 60)))
        return decimal_value.adjusted()


def _scientific_text(mantissa: Decimal, exponent: int) -> str:
    return f"{format(mantissa, 'f').replace('.', ',')} · 10{str(exponent).translate(_SUPERSCRIPT_OUT)}"


def _scientific_form(value: sp.Expr, places: int | None) -> str | None:
    """'4,5 · 10⁴' for 45000; rounded when asked; None when no exact finite form exists."""
    if places is not None:
        rounded = _round_decimal(value, places)
        if rounded == 0:
            return None
        exponent = rounded.adjusted()
        return _scientific_text(rounded.scaleb(-exponent), exponent)
    exact = _decimal_of(value)
    if exact is None:
        return None
    exponent = exact.adjusted()
    return _scientific_text(exact.scaleb(-exponent).normalize(), exponent)


def _comma_number(text: str) -> sp.Rational:
    return sp.Rational(text.replace(",", "."))


def _verify_scientific_notation(input_value: str, submitted_answer: str | None,
                                task_instruction: str | None = None) -> dict[str, Any]:
    check_type = "scientific_notation"
    ok, reason = _preflight(input_value)
    if not ok:
        return _unverifiable(check_type, input_value, _MathInputError(reason or "disallowed_characters_or_syntax"))
    try:
        value = _numeric_value(input_value)
        places = _parse_rounding_places(task_instruction)
        if value == 0:
            raise _MathInputError("unsupported_math_form")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)

    exponent_expected = _power_of_ten(value)
    fields: dict[str, Any] = {"exact_value": _serialize_exact(value), "parser_invoked": True}
    form = _scientific_form(value, places)
    if form is not None:
        fields["scientific_form"] = form
    if submitted_answer is None or not str(submitted_answer).strip():
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"{_REASON_NL['no_claim_to_check']}"
                                      + (f" In wetenschappelijke notatie: {form}." if form else ""))
    fields["submitted_answer"] = submitted_answer
    expected_text = f" Juist is {form}." if form else ""

    def same(candidate: sp.Expr) -> bool:
        return _rounded_compare(value, candidate, places, "")[0]

    answer_text = _expand_superscripts(submitted_answer.replace("−", "-").replace("–", "-"))
    calculator = _CALCULATOR_FORM_RE.fullmatch(answer_text)
    if calculator:
        mantissa, exponent = _comma_number(calculator.group(1)), int(calculator.group(2))
        fields["value_correct"] = same(mantissa * sp.Integer(10) ** exponent)
        fields["misconception"] = "calculator_notation"
        fields["explanation_nl"] = ("Dit is de notatie van de rekenmachine (E). Schrijf a · 10^n, "
                                    f"bijvoorbeeld 4,5 · 10⁴.{expected_text}")
        return _result("wrong", check_type, input_value, **fields)

    scientific = _SCI_FORM_RE.fullmatch(answer_text)
    if scientific:
        mantissa, exponent = _comma_number(scientific.group(1)), int(scientific.group(2))
        if abs(exponent) > MAX_POWER_ABS:
            return _unverifiable(check_type, input_value, _MathInputError("unsupported_math_form"), parser_invoked=True)
        answer_value = mantissa * sp.Integer(10) ** exponent
        normalized = 1 <= abs(mantissa) < 10
    else:
        try:
            answer_value = _numeric_value(submitted_answer)
        except _MathInputError as exc:
            return _unverifiable(check_type, input_value, exc, parser_invoked=True)
        mantissa, exponent = None, None
        normalized = 1 <= abs(answer_value) < 10  # 4,5 is 4,5 · 10⁰

    correct, rounding_fields = _rounded_compare(value, answer_value, places, "")
    fields.update(rounding_fields)
    fields["value_correct"] = correct
    if correct and normalized:
        fields["explanation_nl"] = f"Het antwoord klopt en staat in wetenschappelijke notatie.{expected_text}"
        return _result("verified", check_type, input_value, **fields)
    if correct and scientific:
        fields["misconception"] = "not_normalized"
        fields["explanation_nl"] = ("De waarde klopt, maar het getal voor de macht van 10 moet tussen 1 en 10 liggen."
                                    f"{expected_text}")
        return _result("wrong", check_type, input_value, **fields)
    if correct:
        fields["misconception"] = "not_scientific_notation"
        fields["explanation_nl"] = f"De waarde klopt, maar het antwoord staat niet in wetenschappelijke notatie.{expected_text}"
        return _result("wrong", check_type, input_value, **fields)

    explanation = f"Het antwoord klopt niet.{expected_text}"
    if scientific and normalized and same(mantissa * sp.Integer(10) ** exponent_expected):
        if abs(exponent - exponent_expected) == 1:
            fields["misconception"] = "exponent_off_by_one"
            explanation = f"Het getal voor de macht klopt, maar de exponent is er één naast.{expected_text}"
        elif exponent == -exponent_expected and exponent != 0:
            fields["misconception"] = "exponent_sign_wrong"
            explanation = ("Het getal voor de macht klopt, maar het teken van de exponent is fout: "
                           f"bij een getal kleiner dan 1 is de exponent negatief.{expected_text}")
    fields["explanation_nl"] = explanation
    return _result("wrong", check_type, input_value, **fields)


# --- Divisors, multiples and primes ------------------------------------------
_NUMBER_TASKS = {
    "ggd": "gcd", "grootste gemene deler": "gcd",
    "kgv": "lcm", "kleinste gemene veelvoud": "lcm", "kleinste gemeenschappelijke veelvoud": "lcm",
    "priemfactoren": "factor", "ontbind in priemfactoren": "factor", "priemfactorontbinding": "factor",
    "delers": "divisors", "alle delers": "divisors",
    "priemgetal": "prime", "is priemgetal": "prime",
}
_NUMBER_INPUT_RE = re.compile(r"\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*", re.DOTALL)
_PRIME_FACTOR_RE = re.compile(r"(\d+)(?:\^\(?(\d+)\)?)?")
MAX_NUMBER_THEORY = 10 ** 12


def _whole_numbers(text: str) -> list[int]:
    items = text.split(";") if ";" in text else re.split(r",\s+|\s+en\s+", text.strip())
    numbers: list[int] = []
    for item in items:
        item = item.strip()
        if "." in item:
            raise _MathInputError("ambiguous_dot_notation")
        if not re.fullmatch(r"\d+", item):
            raise _MathInputError("invalid_number_input")
        numbers.append(int(item))
    return numbers


def _factorization_text(n: int) -> str:
    parts = []
    for prime, power in sorted(sp.factorint(n).items()):
        parts.append(str(prime) + (str(power).translate(_SUPERSCRIPT_OUT) if power > 1 else ""))
    return " · ".join(parts)


def _verify_number_theory(input_value: str, submitted_answer: str | None) -> dict[str, Any]:
    check_type = "number_theory"
    try:
        if not isinstance(input_value, str) or len(input_value) > MAX_INPUT_LENGTH:
            raise _MathInputError("invalid_number_input")
        match = _NUMBER_INPUT_RE.fullmatch(input_value)
        if not match:
            raise _MathInputError("invalid_number_input")
        task = _NUMBER_TASKS.get(" ".join(match.group(1).casefold().split()))
        if task is None:
            raise _MathInputError("unsupported_number_task")
        numbers = _whole_numbers(match.group(2))
        needed_many = task in ("gcd", "lcm")
        if (needed_many and len(numbers) < 2) or (not needed_many and len(numbers) != 1):
            raise _MathInputError("invalid_number_input")
        if any(n < 1 or n > MAX_NUMBER_THEORY for n in numbers) or (task == "factor" and numbers[0] < 2):
            raise _MathInputError("invalid_number_input")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc)

    fields: dict[str, Any] = {"task": task, "parser_invoked": True}
    n = numbers[0]
    if task == "gcd":
        exact: Any = int(sp.gcd(numbers))
        fields["exact_value"] = str(exact)
    elif task == "lcm":
        exact = int(sp.lcm(numbers))
        fields["exact_value"] = str(exact)
    elif task == "factor":
        fields["exact_value"] = _factorization_text(n)
    elif task == "divisors":
        exact = [int(d) for d in sp.divisors(n)]
        fields["exact_value"] = ", ".join(str(d) for d in exact)
    else:
        exact = bool(sp.isprime(n))
        fields["exact_value"] = "ja" if exact else "nee"
        if not exact and n > 1:
            fields["factorization"] = _factorization_text(n)

    if submitted_answer is None or not str(submitted_answer).strip():
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"{_REASON_NL['no_claim_to_check']} Het antwoord is {fields['exact_value']}.")
    fields["submitted_answer"] = submitted_answer
    answer_text = submitted_answer.strip()

    def invalid() -> dict[str, Any]:
        return _unverifiable(check_type, input_value, _MathInputError("invalid_answer"), parser_invoked=True)

    if task in ("gcd", "lcm"):
        answer_text = re.sub(r"^(ggd|kgv)\s*=\s*", "", answer_text, flags=re.IGNORECASE)
        if not re.fullmatch(r"\d+", answer_text):
            return invalid()
        given = int(answer_text)
        if given == exact:
            fields["explanation_nl"] = f"Het antwoord klopt: {fields['exact_value']}."
            return _result("verified", check_type, input_value, **fields)
        gcd_value, lcm_value = int(sp.gcd(numbers)), int(sp.lcm(numbers))
        product = 1
        for number in numbers:
            product *= number
        if task == "gcd" and given == lcm_value:
            fields["misconception"] = "confused_ggd_kgv"
            explanation = "Dit is het kgv (kleinste gemene veelvoud), gevraagd is de ggd (grootste gemene deler)."
        elif task == "gcd" and given > 0 and all(number % given == 0 for number in numbers):
            fields["misconception"] = "common_divisor_not_greatest"
            explanation = f"{given} is wel een gemeenschappelijke deler, maar niet de grootste."
        elif task == "lcm" and given == product:
            fields["misconception"] = "product_instead_of_kgv"
            explanation = "Het product is wel een gemeenschappelijk veelvoud, maar niet het kleinste."
        elif task == "lcm" and given == gcd_value:
            fields["misconception"] = "confused_ggd_kgv"
            explanation = "Dit is de ggd (grootste gemene deler), gevraagd is het kgv (kleinste gemene veelvoud)."
        elif task == "lcm" and given > 0 and given % lcm_value == 0:
            fields["misconception"] = "common_multiple_not_least"
            explanation = f"{given} is wel een gemeenschappelijk veelvoud, maar niet het kleinste."
        else:
            explanation = "Het antwoord klopt niet."
        fields["explanation_nl"] = f"{explanation} Het juiste antwoord is {fields['exact_value']}."
        return _result("wrong", check_type, input_value, **fields)

    if task == "factor":
        factors = re.split(r"\s*[·×*]\s*", _expand_superscripts(answer_text))
        product, all_prime = 1, True
        for factor in factors:
            factor_match = _PRIME_FACTOR_RE.fullmatch(factor.strip())
            if not factor_match:
                return invalid()
            base = int(factor_match.group(1))
            power = int(factor_match.group(2) or 1)
            if power > 64 or base > MAX_NUMBER_THEORY:
                return invalid()
            product *= base ** power
            all_prime = all_prime and bool(sp.isprime(base))
        fields["value_correct"] = product == n
        if product == n and all_prime:
            fields["explanation_nl"] = f"Het antwoord klopt: {n} = {fields['exact_value']}."
            return _result("verified", check_type, input_value, **fields)
        if product == n:
            fields["misconception"] = "factor_not_prime"
            fields["explanation_nl"] = ("Het product klopt, maar niet alle factoren zijn priemgetallen. "
                                        f"Ontbind verder: {n} = {fields['exact_value']}.")
        else:
            fields["explanation_nl"] = f"Het product van de factoren is {product}, niet {n}. Juist is {fields['exact_value']}."
        return _result("wrong", check_type, input_value, **fields)

    if task == "divisors":
        try:
            given_list = _whole_numbers(answer_text)
        except _MathInputError as exc:
            if exc.reason == "invalid_number_input" and re.search(r"\d,\d", answer_text):
                exc = _MathInputError("ambiguous_comma_notation")
            elif exc.reason == "invalid_number_input":
                exc = _MathInputError("invalid_answer")
            return _unverifiable(check_type, input_value, exc, parser_invoked=True)
        given_set = set(given_list)
        missing = sorted(set(exact) - given_set)
        extra = sorted(given_set - set(exact))
        if not missing and not extra:
            fields["explanation_nl"] = f"Het antwoord klopt: de delers van {n} zijn {fields['exact_value']}."
            return _result("verified", check_type, input_value, **fields)
        if missing:
            fields["missing_divisors"] = missing
        if extra:
            fields["not_divisors"] = extra
        hint = " Vergeet 1 en het getal zelf niet." if (1 in missing or n in missing) else ""
        fields["explanation_nl"] = f"Het antwoord klopt niet.{hint} De delers van {n} zijn {fields['exact_value']}."
        return _result("wrong", check_type, input_value, **fields)

    lowered = " ".join(answer_text.casefold().split())
    if re.search(r"\b(nee|geen|niet)\b", lowered):
        claims_prime = False
    elif re.search(r"\b(ja|wel)\b", lowered) or "is een priemgetal" in lowered:
        claims_prime = True
    else:
        return invalid()
    if claims_prime == exact:
        fields["explanation_nl"] = (f"Het antwoord klopt: {n} is {'een' if exact else 'geen'} priemgetal"
                                    + (f" ({n} = {fields['factorization']})." if "factorization" in fields else "."))
        return _result("verified", check_type, input_value, **fields)
    if n == 1:
        explanation = "1 is geen priemgetal: een priemgetal heeft precies twee delers."
    elif exact:
        explanation = f"{n} is wel een priemgetal: het heeft alleen 1 en {n} als delers."
    else:
        explanation = f"{n} is geen priemgetal: {n} = {fields['factorization']}."
    fields["explanation_nl"] = explanation
    return _result("wrong", check_type, input_value, **fields)


# --- Counting: order and replacement ------------------------------------------
MAX_COUNTING_N = 1000
MAX_COUNTING_K = 100


def _counting_spec(spec: str) -> tuple[bool, bool]:
    """(ordered, with_replacement) from Dutch words; never guessed."""
    text = " ".join(spec.casefold().split())
    if "zonder volgorde" in text or "volgorde niet belangrijk" in text or "volgorde is niet belangrijk" in text:
        ordered: bool | None = False
    elif "met volgorde" in text or "volgorde belangrijk" in text or "volgorde is belangrijk" in text:
        ordered = True
    else:
        ordered = None
    if "zonder terugleggen" in text:
        replacement: bool | None = False
    elif "met terugleggen" in text:
        replacement = True
    else:
        replacement = None
    if ordered is None or replacement is None:
        raise _MathInputError("ambiguous_counting")
    return ordered, replacement


def _verify_counting(input_value: str, submitted_answer: str | None) -> dict[str, Any]:
    check_type = "counting"
    try:
        if not isinstance(input_value, str) or ":" not in input_value or len(input_value) > MAX_INPUT_LENGTH:
            raise _MathInputError("invalid_counting_input")
        spec, numbers_text = input_value.rsplit(":", 1)
        ordered, replacement = _counting_spec(spec)
        if not ordered and replacement:
            # Unordered with replacement (n + k - 1 boven k) is not school material.
            raise _MathInputError("unsupported_counting")
        numbers = _whole_numbers(numbers_text)
        if len(numbers) != 2:
            raise _MathInputError("invalid_counting_input")
        n, k = numbers
        if n < 1 or n > MAX_COUNTING_N or k > MAX_COUNTING_K or (not replacement and k > n):
            raise _MathInputError("invalid_counting_input")
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc)

    ordered_with = sp.Integer(n) ** k
    ordered_without = sp.factorial(n) / sp.factorial(n - k) if k <= n else None
    unordered_without = sp.binomial(n, k) if k <= n else None
    if ordered and replacement:
        exact, formula = ordered_with, f"{n}^{k}"
    elif ordered:
        exact, formula = ordered_without, f"{n}!/({n} - {k})!"
    else:
        exact, formula = unordered_without, f"{n} boven {k}"
    fields: dict[str, Any] = {"exact_value": _serialize_exact(exact), "formula": formula, "parser_invoked": True}

    if submitted_answer is None or not str(submitted_answer).strip():
        return _result("unverifiable", check_type, input_value, **fields, reason="no_claim_to_check",
                       explanation_nl=f"{_REASON_NL['no_claim_to_check']} Het aantal is {formula} = {fields['exact_value']}.")
    fields["submitted_answer"] = submitted_answer
    try:
        answer = _numeric_value(submitted_answer)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)

    if sp.simplify(answer - exact) == 0:
        fields["explanation_nl"] = f"Het antwoord klopt: {formula} = {fields['exact_value']}."
        return _result("verified", check_type, input_value, **fields)

    candidates: list[tuple[str, sp.Expr | None, str]] = []
    if ordered and not replacement:
        candidates = [("with_replacement_used", ordered_with,
                       "Zo tel je met terugleggen; hier kan elk ding maar één keer gekozen worden."),
                      ("order_ignored", unordered_without,
                       "Zo tel je zonder volgorde; hier is de volgorde wel belangrijk.")]
    elif ordered and replacement:
        candidates = [("without_replacement_used", ordered_without,
                       "Zo tel je zonder terugleggen; hier mag elke keuze opnieuw gekozen worden.")]
    else:
        candidates = [("order_counted", ordered_without,
                       "Zo tel je met volgorde; hier is de volgorde niet belangrijk, dus deel nog door k!."),
                      ("with_replacement_used", ordered_with,
                       "Zo tel je met volgorde en met terugleggen; hier geen van beide.")]
    for name, value, explanation in candidates:
        if value is not None and value != exact and sp.simplify(answer - value) == 0:
            fields["misconception"] = name
            fields["explanation_nl"] = f"{explanation} Juist is {formula} = {fields['exact_value']}."
            return _result("wrong", check_type, input_value, **fields)
    fields["explanation_nl"] = f"Het antwoord klopt niet. Juist is {formula} = {fields['exact_value']}."
    return _result("wrong", check_type, input_value, **fields)


# Geometry lives in its own package; verify_math forwards these check types to it.
GEOMETRY_CHECK_TYPES = frozenset({
    "angle_sum", "shape_formula", "scale_factor", "trig_ratio",
    "sine_rule", "cosine_rule", "point_distance", "midpoint",
})


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
            unit_conversion, percentage_change, inequality_solution, system_solution,
            function_notation, statistics, scientific_notation, number_theory, counting,
            and the geometry check types in GEOMETRY_CHECK_TYPES.
        input: expression, equation or quantity; not used to infer a word problem's meaning.
        submitted_answer: the student's answer to compare, exactly as written.
        steps: explicit equation steps for step_validation. When omitted, an
            arrow-delimited input string is accepted for compatibility.
        task_instruction: optional Dutch rounding instruction, e.g.
            ``Rond af op 2 decimalen.`` or ``Rond af op 1 decimaal.``
        variable: the letter to solve for when an equation contains more than
            one letter (equation_solution, step_validation, inequality_solution,
            function_notation).
        substitutions: values to fill in for formula_substitution, e.g.
            ``{"p": "3"}``; for geometry check types these are the sizes,
            e.g. ``{"r": "3"}`` or ``{"A": "30°", "schuine zijde": "10"}``.

    Returns a dictionary with status in {verified, wrong, unverifiable}.
    """
    allowed_types = {"arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution",
                     "step_validation", "formula_substitution", "formula_rewrite", "unit_conversion",
                     "percentage_change", "inequality_solution", "system_solution", "function_notation",
                     "statistics", "scientific_notation", "number_theory", "counting"}
    if check_type in GEOMETRY_CHECK_TYPES:
        # Imported here: geometry imports helpers from this module.
        from geometry import verify_geometry
        return verify_geometry(check_type, input, submitted_answer, task_instruction, values=substitutions)
    if check_type not in allowed_types:
        return _unverifiable(str(check_type), input, _MathInputError("unsupported_math_form"))
    if not isinstance(input, str):
        return _unverifiable(check_type, input, _MathInputError("disallowed_characters_or_syntax"))

    if check_type == "arithmetic":
        return _verify_arithmetic(check_type, input, submitted_answer, task_instruction)
    if check_type == "answer_equivalence":
        return _verify_answer_equivalence(input, submitted_answer, task_instruction)
    if check_type == "percentage_change":
        return _verify_percentage_change(input, submitted_answer, task_instruction)
    if check_type == "expression_equivalence":
        return _verify_expression_equivalence(input)
    if check_type == "equation_solution":
        return _verify_equation_solution(input, submitted_answer, task_instruction, variable)
    if check_type == "formula_substitution":
        return _verify_formula_substitution(input, substitutions, submitted_answer, task_instruction)
    if check_type == "formula_rewrite":
        return _verify_formula_rewrite(input, submitted_answer)
    if check_type == "unit_conversion":
        return _verify_unit_conversion(input, submitted_answer, task_instruction)
    if check_type == "inequality_solution":
        return _verify_inequality_solution(input, submitted_answer, task_instruction, variable)
    if check_type == "system_solution":
        return _verify_system_solution(input, submitted_answer, task_instruction)
    if check_type == "function_notation":
        return _verify_function_notation(input, submitted_answer, task_instruction, variable)
    if check_type == "statistics":
        return _verify_statistics(input, submitted_answer, task_instruction)
    if check_type == "scientific_notation":
        return _verify_scientific_notation(input, submitted_answer, task_instruction)
    if check_type == "number_theory":
        return _verify_number_theory(input, submitted_answer)
    if check_type == "counting":
        return _verify_counting(input, submitted_answer)
    return _verify_steps(input, steps, variable)


if __name__ == "__main__":
    # Tiny CLI for manual smoke checks; not a production MCP transport.
    import argparse
    import json

    cli = argparse.ArgumentParser(description="Conservative Dutch maths verifier")
    cli.add_argument("check_type", choices=["arithmetic", "answer_equivalence", "expression_equivalence", "equation_solution",
                                            "step_validation", "formula_substitution", "formula_rewrite", "unit_conversion",
                                            "percentage_change", "inequality_solution", "system_solution",
                                            "function_notation", "statistics", "scientific_notation",
                                            "number_theory", "counting", *sorted(GEOMETRY_CHECK_TYPES)])
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
