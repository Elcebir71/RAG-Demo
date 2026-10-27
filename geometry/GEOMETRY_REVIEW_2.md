# Second review of verify_geometry (27 Oct 2026)

Scope: `geometry/verify_geometry.py` (1962 lines) after the first review, GEO001–GEO105.

## Fixed — well done
- No `sympify`, `eval` or `parse_expr` on user text anymore; student answers go through `_safe_parse_expr`.
- Correct answer is checked before misconceptions in all check types; a misconception whose value
  equals the exact value is skipped (45°, r = 2, k = 1).
- `_result(...)` helper: every result has `status`, `check_type`, `input`, `submitted_answer`,
  `parser_invoked`. G6 now fails closed (`unverifiable`) on an unreadable answer.
- The duplicate `_serialize_exact` is gone.
- Units: mixed units give `unit_mismatch`; cm with an m² answer gives `wrong_unit`.
- Sine rule: obtuse known angle with a longer other side gives `impossible_triangle`.

## Still to fix (3 items, then the module can be integrated)

### 1. Replace `_handle_rounding_comparison` with `_rounded_compare` directly
`_rounded_compare` returns a tuple `(matched: bool, fields: dict)`. The helper ignores that dict,
recomputes the rounded value through a `float` with 15 digits, has dead branches for return types
that never occur, and falls back to a string comparison when an exception is swallowed.
It also breaks with significant figures: verify_math now supports
"Rond af op 3 significante cijfers." (`_parse_rounding_places` returns an int subclass), and the
helper would treat 3 significant figures as 3 decimals and write `decimal_places`.
Replace every call with:

```python
matched, rounding_fields = _rounded_compare(exact, submitted_expr, places, answer_text)
if matched:
    return _result("verified", ..., exact_value=exact_value_str, **rounding_fields,
                   explanation_nl="Correct afgerond.")
```

`rounding_fields` already holds `rounded_answer` and either `decimal_places` or
`significant_figures`. Delete `_handle_rounding_comparison` and `_compute_half_up_str`.
Hakan will first refresh the read-only copy of `verify_math.py` from RAG-Demo (it now has
significant figures).

### 2. Coordinates: unexpected input must never crash
In `_parse_coordinate_pair`, the semicolon branch calls `sp.Rational(...)` without a try.
`"A(a; 2), B(4; 6)"` raises a `TypeError`, which `_verify_point_distance` does not catch
(it only catches `ValueError`), so the call crashes instead of returning `unverifiable`.
Fix: parse each coordinate with `_numeric_value(part)` (decimal comma, fractions, negative numbers
and √ all work) and turn any `_MathInputError` into `unverifiable`.

### 3. Units in `scale_factor`
`origineel: "5 cm²"` with answer `"20 m²"` is still `verified`; only `shape_formula` compares the
answer unit. Apply the same rule: the answer unit must equal the unit of `origineel`, otherwise
`wrong` with `misconception: wrong_unit`.

## Small things
- Explanations: "Misconceptie gevonden: forgot_half_triangle" shows a code name to the teacher.
  Write one Dutch sentence per misconception (e.g. "De oppervlakte van een driehoek is ½ · basis · hoogte;
  de ½ is vergeten.").
- "Correct afgerond op {places} decimalen." is wrong for significant figures; use a neutral sentence.
- `description` in `geometry_cases.yaml` still says G1–G5; GEO058–GEO068 still have
  `curriculum_scope: vwo1_candidate` (should be `vwo4_wiskunde_b`).
- Unused imports: `math`, `Dict`, `List`, `Optional`, `Tuple`.
- Definition of done: add the progress notes at the end of `GEOMETRY_TASKS.md` (what was built per
  task, which cases cover it, open questions).

## New test cases (add first, see them fail)

```yaml
  - id: GEO106
    title: Unreadable coordinate never crashes
    check_type: point_distance
    curriculum_scope: tool_boundary_test
    input: "A(a; 2), B(4; 6)"
    submitted_answer: "5"
    expected: {status: unverifiable}

  - id: GEO107
    title: Distance rounded to significant figures
    check_type: point_distance
    curriculum_scope: vwo1_candidate
    input: "A(0, 0), B(1, 1)"
    submitted_answer: "1,4"
    task_instruction: "Rond af op 2 significante cijfers."
    expected: {status: verified, rounded_answer: "1,4", significant_figures: 2}

  - id: GEO108
    title: Scale factor answer unit must match the original unit
    check_type: scale_factor
    curriculum_scope: vwo1_candidate
    input: "oppervlakte"
    values: {k: "2", origineel: "5 cm²"}
    submitted_answer: "20 m²"
    expected: {status: wrong, misconception: wrong_unit}

  - id: GEO109
    title: Coordinates may be fractions
    check_type: midpoint
    curriculum_scope: vwo1_candidate
    input: "A(1/2; 3), B(5/2; 5)"
    submitted_answer: "(3/2; 4)"
    expected: {status: verified}
```

GEO107 only passes after the copy of `verify_math.py` has been refreshed. GEO109 probably passes
already; it guards that fractions keep working after the switch to `_numeric_value`.
