# Review of verify_geometry G1–G6 (27 Oct 2026)

Reviewer: Claude, for Hakan. Scope: `geometry/verify_geometry.py` (2296 lines),
`tests/geometry/test_geometry.py`, `tests/geometry/geometry_cases.yaml` (GEO001–GEO095).

The test cases are mathematically correct (every expected value was recalculated by hand).
The test runner is fine (`parents[2]`, optional keys passed only when present).
The code below needs fixes before integration. Work test-first: add the new cases in
section 3 first, see them fail, then fix.

## 1. Must fix (blocking)

### 1.1 `sp.sympify` on user text (contract rule 4: security)
- `_parse_submitted_distance` (fallback, ~line 2042) and `_parse_submitted_point` (~line 2062)
  call `sp.sympify` on the student's answer. `sympify` evaluates Python code, so an answer like
  `__import__('os').getcwd()` is executed.
- Fix: delete both `sympify` paths. Use `_safe_parse_expr` / `_numeric_value` for every value.
  `√2`, `6√2` and `2,5` are already handled by `_safe_parse_expr`, so no fallback is needed.
  If parsing fails: `unverifiable` with the parser's reason (see 1.3).

### 1.2 Misconceptions are checked before the correct answer
In `_verify_shape_formula`, `_verify_scale_factor` and `_verify_trig_ratio`, the
misconception loop runs before the "is it correct?" check. When the misconception value equals
the correct value, a correct answer is reported as `wrong`:
- `sin(A)` with A = 45°: sin 45° = cos 45°, so the right answer `5√2` becomes `swapped_sin_cos`.
- Circle with r = 2: circumference 4π = area 4π, so `4π` becomes `swapped_circumference_and_area`.
- Scale factor k = 1: k = k² = k³.
Fix: check correctness first; only test misconceptions when the answer is wrong. Also skip a
misconception whose value equals the exact value.

### 1.3 G6 (`point_distance`, `midpoint`) does not follow the contract
- Results miss `check_type`, `input`, `submitted_answer`, `parser_invoked`, and often `explanation_nl`.
  Every result must have the same shape as verify_math (contract section 3).
- An answer that cannot be parsed returns `wrong` (~lines 2126, 2215). Contract rule 3 (fail closed):
  that must be `unverifiable` with a reason (e.g. `invalid_answer`).
- Own rounding code: `re.search(r"Rond af op (\d+) decimal", ...)` does not match
  "Rond af op 1 decimaal." ("decimaal" does not contain "decimal"), so a 1-decimal instruction is
  ignored. Use `_parse_rounding_places` and `_rounded_compare` (already imported, never used).
- A second `_serialize_exact` is defined at module level (~line 2064). Because it comes after the
  import, it replaces verify_math's version for the WHOLE module (G1–G5 too). Delete it.

## 2. Should fix

### 2.1 Rounding (G2–G5)
`round(exact_float, places) == sub_float` uses binary floats and Python's round, not
half-up. Example: scale factor 1,5 × 1,7 = 2,55 exactly; half-up to 1 decimal gives 2,6, but
`round(2.55, 1)` gives 2.5, so the correct student answer 2,6 is marked wrong.
Also `rounded_answer` echoes the student's text instead of the computed rounded value.
Fix: use `_rounded_compare(exact, answer, places, answer_text)` everywhere; it returns
`decimal_places` and the computed `rounded_answer`. Significant figures
("Rond af op 3 significante cijfers.") will then work automatically once verify_math supports them.

### 2.2 Units in `shape_formula` and `scale_factor`
- Units in `values` are stripped and ignored. `{l: "8 m", b: "30 cm"}` gives 240 without
  conversion. Fix: return `unverifiable`, `reason: unit_mismatch`, when the value units differ
  (or convert with one fixed table, but then add tests).
- Only the dimension of the answer unit is checked: `{l: "8 cm", b: "3 cm"}` with answer
  `"24 m²"` is `verified`. The answer unit must be the matching unit (cm → cm²), otherwise
  `wrong` with `misconception: wrong_unit`.

### 2.3 Sine rule with an obtuse known angle
If the known angle is obtuse and the other side is longer, no triangle exists; the code reports
`ambiguous_triangle`. It should be `impossible_triangle`.

## 3. New test cases (add first, they must fail before the fix)

```yaml
  - id: GEO096
    title: Correct answer at 45 degrees is not a sin/cos swap
    check_type: trig_ratio
    curriculum_scope: vwo1_candidate
    input: "sin(A) = overstaande / schuine zijde"
    values: {A: "45°", "schuine zijde": "10"}
    submitted_answer: "5√2"
    expected: {status: verified, exact_value: "5*sqrt(2)"}

  - id: GEO097
    title: Circle with r = 2, circumference equals area numerically
    check_type: shape_formula
    curriculum_scope: vwo1_candidate
    input: "omtrek cirkel"
    values: {r: "2"}
    submitted_answer: "4π"
    expected: {status: verified, exact_value: "4*pi"}

  - id: GEO098
    title: Code in an answer is never executed (distance)
    check_type: point_distance
    curriculum_scope: tool_boundary_test
    input: "A(1, 2), B(4, 6)"
    submitted_answer: "__import__('os').getcwd()"
    expected: {status: unverifiable, parser_invoked: false}

  - id: GEO099
    title: Code in an answer is never executed (midpoint)
    check_type: midpoint
    curriculum_scope: tool_boundary_test
    input: "A(1, 2), B(5, 6)"
    submitted_answer: "(__import__('os').getcwd(), 1)"
    expected: {status: unverifiable}

  - id: GEO100
    title: Distance rounded to 1 decimaal
    check_type: point_distance
    curriculum_scope: vwo1_candidate
    input: "A(0, 0), B(1, 1)"
    submitted_answer: "1,4"
    task_instruction: "Rond af op 1 decimaal."
    expected: {status: verified, rounded_answer: "1,4", decimal_places: 1}

  - id: GEO101
    title: Half-up rounding (2,55 to 1 decimal is 2,6)
    check_type: scale_factor
    curriculum_scope: vwo1_candidate
    input: "lengte"
    values: {k: "1,5", origineel: "1,7"}
    submitted_answer: "2,6"
    task_instruction: "Rond af op 1 decimaal."
    expected: {status: verified, rounded_answer: "2,6"}

  - id: GEO102
    title: Unparseable midpoint answer is unverifiable, not wrong
    check_type: midpoint
    curriculum_scope: tool_boundary_test
    input: "A(1, 2), B(5, 6)"
    submitted_answer: "het midden"
    expected: {status: unverifiable}

  - id: GEO103
    title: Every result has the standard fields
    check_type: point_distance
    curriculum_scope: vwo1_candidate
    input: "A(1, 2), B(4, 6)"
    submitted_answer: "5"
    expected: {status: verified, check_type: point_distance, input: "A(1, 2), B(4, 6)", submitted_answer: "5", parser_invoked: true}

  - id: GEO104
    title: Mixed units in the values are not silently combined
    check_type: shape_formula
    curriculum_scope: tool_boundary_test
    input: "oppervlakte rechthoek"
    values: {l: "8 m", b: "30 cm"}
    submitted_answer: "2,4 m²"
    expected: {status: unverifiable, reason: unit_mismatch}

  - id: GEO105
    title: Answer unit must match the value unit (cm gives cm²)
    check_type: shape_formula
    curriculum_scope: vwo1_candidate
    input: "oppervlakte rechthoek"
    values: {l: "8 cm", b: "3 cm"}
    submitted_answer: "24 m²"
    expected: {status: wrong, misconception: wrong_unit}
```

## 4. Small things
- `curriculum_scope` for sine_rule / cosine_rule (GEO058–GEO068): this is wiskunde B (vwo 4),
  use `vwo4_wiskunde_b` instead of `vwo1_candidate`.
- Unused imports: `math`, `_numeric_value`, `_rounded_compare` (the last one should be used, see 2.1).
- The ~60 repeated result dictionaries make the file 2296 lines. A local helper
  `_result(status, check_type, input, **fields)` would cut this roughly in half and makes it
  impossible to forget a field (the cause of 1.3).
- `values.get("l") or values.get("lengte")`: a value of 0 is falsy and silently falls through.
  Use an explicit `is None` check.

## 5. Order of work
1. Add GEO096–GEO105, run, see them fail.
2. Fix 1.1 (sympify) and 1.2 (order), then 1.3 (G6 contract), then 2.1–2.3.
3. Run all geometry tests; then write the progress notes at the end of GEOMETRY_TASKS.md.
4. G7 (Pythagoras with a named right angle) only after this, as agreed.
