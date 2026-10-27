# verify_math helpers — reference for the geometry module

Extracted from the current `verify_math.py` (with roots and percentages, 247 tests passing).
Use these helpers through `from verify_math import ...`. Do not change them; if one does not fit,
write it down in the progress notes of GEOMETRY_TASKS.md.

## Quick summary

| Helper | Input | Output | Errors |
|---|---|---|---|
| `_MathInputError(reason)` | a reason code | exception with `.reason` and `.message_nl` (Dutch text from `_REASON_NL`) | — |
| `_preflight(text)` | raw user text | `(True, None)` or `(False, reason)` | never raises |
| `_safe_parse_expr(text, allow_variable_denominators=False)` | expression text, Dutch notation | a SymPy expression (may contain letters) | `_MathInputError` (preflight reasons, `variable_denominator_unsupported`, `division_by_zero`, `not_real`, `unsupported_math_form`, ...) |
| `_numeric_value(text)` | text that must be a number | exact SymPy number, simplified (e.g. `sqrt(41)`, `9*pi`, `7/4`) | `_MathInputError` (`not_numeric_expression` if a letter is left, `unsupported_math_form` for oo/nan/complex, plus all parse reasons) |
| `_parse_rounding_places(instruction)` | task instruction or None | `None` (no rounding asked) or number of decimals 0–8 | `_MathInputError("rounding_instruction_not_supported")` if it says 'rond af' in a form it cannot read (e.g. significant digits) |
| `_rounded_compare(exact, answer, places, answer_text)` | exact value, student's value (both SymPy), places from above, the student's raw answer text | `(correct: bool, fields: dict)`; `fields` has `decimal_places` and `rounded_answer` when rounding was used | never raises for normal numbers |
| `_serialize_exact(expr)` | SymPy expression | string for `exact_value`, e.g. `"sqrt(41)"`, `"9*pi"`, `"125/9"` | — |
| `_result(status, check_type, input_value, **fields)` | — | the result dict in the standard shape | — |
| `_unverifiable(check_type, input_value, error, parser_invoked=False)` | a `_MathInputError` | standard `unverifiable` result with `reason` and `explanation_nl` | — |

## How they fit together (typical check)

```python
from verify_math import (_MathInputError, _numeric_value, _parse_rounding_places,
                         _rounded_compare, _serialize_exact, _result, _unverifiable)

def _verify_angle_sum_example(input_value, submitted_answer, task_instruction):
    check_type = "angle_sum"
    try:
        exact = ...                                  # your own computation, a SymPy number
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    fields = {"exact_value": _serialize_exact(exact), "parser_invoked": True}
    if submitted_answer is None:                     # contract rule 7
        return _result("unverifiable", check_type, input_value, **fields,
                       reason="no_claim_to_check", explanation_nl="...")
    try:
        answer = _numeric_value(submitted_answer)    # Dutch notation, √, π, %
        places = _parse_rounding_places(task_instruction)
    except _MathInputError as exc:
        return _unverifiable(check_type, input_value, exc, parser_invoked=True)
    correct, rounding_fields = _rounded_compare(exact, answer, places, submitted_answer)
    fields.update(rounding_fields)
    return _result("verified" if correct else "wrong", check_type, input_value, **fields,
                   submitted_answer=submitted_answer, explanation_nl="...")
```

Notes:
- `_numeric_value` does **not** read units or `°`. Strip them yourself first (e.g. `"73°"` → `"73"`,
  `"28 cm²"` → `"28"` plus unit `"cm²"`), then pass the number part.
- `_rounded_compare` scales by 100 only when `answer_text` ends with `%`; for geometry pass the
  number text without units.
- Exact values: SymPy keeps `sqrt(41)` and `pi` exact; never convert to float before comparing.
- New reason codes for geometry (e.g. `impossible_angles`, `ambiguous_triangle`) are fine; give
  your `_MathInputError` a Dutch message with `_MathInputError("impossible_angles", "...")`,
  because `_REASON_NL` lives in `verify_math.py` and must not be edited.

## Source code (read-only)

### _MathInputError and _REASON_NL
```python
class _MathInputError(ValueError):
    def __init__(self, reason: str, message_nl: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.message_nl = message_nl or _REASON_NL.get(reason, "Deze invoer kan niet betrouwbaar worden gecontroleerd.")
```

```python
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
}
```

### _preflight
```python
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
```

### _safe_parse_expr
```python
def _safe_parse_expr(text: Any, allow_variable_denominators: bool = False) -> sp.Expr:
    ok, reason = _preflight(text)
    if not ok:
        raise _MathInputError(reason or "disallowed_characters_or_syntax")
    denominators: list[sp.Expr] = []
    value = _parse_expr(text, denominators)
    if denominators and not allow_variable_denominators:
        raise _MathInputError("variable_denominator_unsupported")
    return value
```

### _numeric_value
```python
def _numeric_value(text: str) -> sp.Expr:
    value = _safe_parse_expr(text)
    if value.free_symbols or value.is_number is not True:
        raise _MathInputError("not_numeric_expression")
    value = sp.simplify(value)
    if value in (sp.nan, sp.oo, -sp.oo, sp.zoo) or value.is_finite is False or value.is_real is False:
        raise _MathInputError("unsupported_math_form")
    return value
```

### _parse_rounding_places
```python
def _parse_rounding_places(instruction: str | None) -> int | None:
    if not instruction:
        return None
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
```

### _round_decimal, _rounded_rational, _rounded_display (used by _rounded_compare)
```python
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
```

```python
def _rounded_rational(value: sp.Expr, places: int) -> sp.Expr:
    """Return an exact rational representation of the rounded decimal."""
    rounded = _round_decimal(value, places)
    return sp.Rational(str(rounded))
```

```python
def _rounded_display(value: sp.Expr, places: int) -> str:
    """Format the rounded decimal in Dutch notation, preserving requested places."""
    rounded = _round_decimal(value, places)
    text = f"{rounded:.{places}f}"
    return text.replace(".", ",")
```

### _rounded_compare
```python
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
    return matched, {"decimal_places": places, "rounded_answer": display}
```

### _serialize_exact
```python
def _serialize_exact(expr: sp.Expr) -> str:
    expr = sp.simplify(expr)
    return sp.sstr(expr)
```

### _result and _unverifiable
```python
def _result(status: str, check_type: str, input_value: Any, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "check_type": check_type,
        "input": input_value,
    }
    result.update(fields)
    return result
```

```python
def _unverifiable(check_type: str, input_value: Any, error: _MathInputError, *, parser_invoked: bool = False) -> dict[str, Any]:
    return _result(
        "unverifiable", check_type, input_value,
        reason=error.reason,
        explanation_nl=error.message_nl,
        parser_invoked=parser_invoked,
    )
```

