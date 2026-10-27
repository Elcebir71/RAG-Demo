# Geometry module for verify_math — task list

This is a hand-over for a separate contributor (another AI assistant) who builds the geometry part
of `verify_math`. Read the whole document before writing code.

## 1. Context

`verify_math` (repo `Elcebir71/RAG-Demo`) independently checks maths claims for Dutch secondary
school (vwo, starting in onderbouw) with SymPy. An AI or a student gives a claim, the tool answers
with one of three statuses:

| status | meaning |
|---|---|
| `verified` | the mathematical property was checked and holds |
| `wrong` | it was checked and does not hold |
| `unverifiable` | the tool cannot check it reliably; it says why (`reason`) and never guesses |

It runs locally as an MCP server for Claude Desktop. 247 tests pass today. The owner (Hakan, a
maths teacher) runs all tests on his own machine and makes every commit himself.

## 2. Rules you must follow (the contract)

1. **Do not edit `verify_math.py`, `mcp_server.py` or `tests/verify_math_cases.yaml`.** Work only
   in the new files listed in section 3. Integration happens later, in one small step.
2. **Tests first.** For every task, first add the acceptance cases to `tests/geometry/geometry_cases.yaml`,
   have Hakan run them and see them fail, then write the code.
3. **Fail closed.** When input is ambiguous, unsupported or needs a figure, return
   `unverifiable` with a specific `reason`. Never return `verified` by guessing.
4. **No `eval`, no `exec`, no `sympy.parse_expr`, no `sympify` on user text.** Reuse the safe
   parser through the public helper described in section 3, or write a small grammar of your own.
5. **Dutch school notation:** decimal comma (`3,5`), `×` and `·` for multiplication, `:` for
   division, `°` for degrees, `√` for roots, `π` for pi. A period inside a number (`1.000`) is
   ambiguous and must be rejected (`ambiguous_dot_notation`).
6. **Exact by default, rounding only when asked.** An approximate answer (`6,40`) is only correct
   when `task_instruction` asks for rounding ("Rond af op 2 decimalen." / "Rond af op 1 decimaal.").
   Without it, `6,40` for `√41` is `wrong`.
7. **No answer means nothing was checked:** without `submitted_answer`, return `unverifiable`
   with `reason: no_claim_to_check` and still include the computed `exact_value`.
8. **Explanations in Dutch** (`explanation_nl`), short and concrete. Code, names and comments in English.
9. **Figures are out of scope.** The tool never reads a figure or decides which side is the
   hypotenuse or which angles are equal from a drawing. The caller gives the formula or the
   relation; the tool checks the numbers.
10. **No copied textbook content.** Write your own synthetic test cases; do not copy exercises
    from Getal & Ruimte, KERN, Math4All or other methods.

## 3. Files and interface

Workspace: a separate, clean local folder, **not** the RAG-Demo repository. Nothing goes to GitHub
until the module is finished; then Hakan copies it into RAG-Demo in one step.

```
C:\VS-CODE\geometry-lab\        # its own local git repo (git init), no remote
  verify_math.py                 # READ-ONLY copy from RAG-Demo, for the helper imports
  pytest.ini                     # [pytest] testpaths = tests
  geometry\ ...                  # see below
  tests\geometry\ ...            # see below
```

Use the existing virtual environment of RAG-Demo (it has sympy, pyyaml and pytest):
`C:\VS-CODE\rag-demo\.venv\Scripts\Activate.ps1`, then work in `geometry-lab`.

Never edit the copy of `verify_math.py`. If a helper is missing or wrong, write it down in the
progress notes; Hakan decides whether `verify_math.py` changes in RAG-Demo, and then refreshes the copy.

Inside the workspace, all geometry work lives in its own folders:

```
geometry/
  __init__.py                    # exports verify_geometry
  verify_geometry.py             # all geometry checks
  GEOMETRY_TASKS.md              # this file, with your progress notes at the end
tests/geometry/
  test_geometry.py               # runner, copy the style of tests/test_verify_math.py
  geometry_cases.yaml            # acceptance cases, same format as verify_math_cases.yaml
```

Why `tests/geometry/` and not `geometry/tests/`: in RAG-Demo, `pytest.ini` only searches `tests/`
and the GitHub workflow runs plain `python -m pytest -q`. Using the same layout here means the
folders can later be copied into RAG-Demo as they are, without changing any configuration.

Imports: the test file sits two levels below the repository root, so put the root on the path
with `Path(__file__).resolve().parents[2]` (not `parents[1]` as in `tests/test_verify_math.py`),
then `from geometry import verify_geometry`. Inside `geometry/verify_geometry.py`, import the
helpers with `from verify_math import ...`; the repository root is on the path when the MCP server
and pytest run (in geometry-lab that is the read-only copy).

Public function:

```python
def verify_geometry(
    check_type: str,
    input: str = "",
    submitted_answer: str | None = None,
    task_instruction: str | None = None,
    values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Returns the same shape as verify_math: status, check_type, input, reason?, exact_value?,
    explanation_nl, parser_invoked, plus check-specific fields."""
```

Result dictionaries must use the same field names as `verify_math` where they mean the same thing:
`status`, `reason`, `exact_value`, `submitted_answer`, `explanation_nl`, `rounded_answer`,
`decimal_places`, `parser_invoked`, `misconception`.

You may import these helpers from `verify_math` (read-only use; do not change them):
`_safe_parse_expr`, `_numeric_value`, `_parse_rounding_places`, `_rounded_compare`,
`_serialize_exact`, `_MathInputError`. If you need something else, ask Hakan instead of editing
`verify_math.py`.

Later integration (not your job): `verify_math()` will forward the geometry check types to
`verify_geometry()`.

## 4. Tasks, in this order

Each task: about 6–12 acceptance cases, including at least one `wrong` case with a realistic
student mistake (`misconception`) and at least one `unverifiable` case.

### G1 — Degrees and angle sums
- Read `°` (`35°`, `35,5°`).
- `check_type: angle_sum`: input lists the known angles of a triangle or quadrilateral, e.g.
  `"driehoek: 35°, 72°"`, answer `"73°"`. Triangle 180°, quadrilateral 360°, straight angle 180°,
  full angle 360°.
- Reject impossible input (angles that already sum to 180° or more in a triangle) as `unverifiable`
  with `reason: impossible_angles`.
- Misconception example: using 360° for a triangle.

### G2 — Perimeter, area and volume with a named formula
- `check_type: shape_formula`: input names the shape and quantity, values give the sizes, e.g.
  input `"oppervlakte rechthoek"`, values `{"l": "8", "b": "3,5"}`, answer `"28"`.
- Fixed table of shapes: rechthoek, vierkant, driehoek (½ · basis · hoogte), parallellogram,
  trapezium, cirkel (omtrek 2πr, oppervlakte πr²), balk, kubus, cilinder, prisma, piramide, kegel, bol.
- Units are optional: if values and answer carry units (`cm`, `cm²`, `cm³`), check that the answer's
  unit has the right dimension (length / area / volume); a wrong dimension is `wrong` with
  `misconception: wrong_unit_dimension`.
- Misconception examples: driehoek without the ½; omtrek and oppervlakte of a circle swapped.
- Exact answers with π (`9π`) are `verified`; `28,27` only with a rounding instruction.

### G3 — Scale factor (vergroten en verkleinen)
- `check_type: scale_factor`: values give the factor `k` and an original length, area or volume;
  the answer is the new one. Length × k, area × k², volume × k³.
- Misconception: area multiplied by k instead of k².

### G4 — Sine, cosine and tangent in a right triangle (degrees)
- `check_type: trig_ratio`: input names the ratio and what is given, e.g.
  `"sin(A) = overstaande / schuine zijde"`, values `{"A": "30°", "schuine zijde": "10"}`,
  answer `"overstaande = 5"`.
- Also the inverse: a side ratio given, answer an angle in degrees (rounded only when asked).
- Always degrees. Radians are out of scope for now → `unverifiable`.
- Misconception: sine and cosine swapped.

### G5 — Sinusregel and cosinusregel (wiskunde B)
- `check_type: sine_rule` and `cosine_rule`, same style as G4.
- The ambiguous case of the sinusregel (two possible triangles) must be reported, not hidden:
  return both angles or `unverifiable` with `reason: ambiguous_triangle`.

### G6 — Coordinates (analytic geometry)
- `check_type: point_distance` and `midpoint`: input `"A(1, 2), B(4, 6)"`, answer `"5"` or
  `"(2,5; 4)"`.
- Coordinates in Dutch notation use `;` as separator when decimals have a comma: `(2,5; 4)`.
  A point like `(1,5)` without `;` stays ambiguous → `unverifiable`, `ambiguous_comma_notation`.

## 5. Out of scope (return `unverifiable`, do not build)

- Reading or interpreting a figure, a drawing or a photo.
- Proofs ("toon aan dat ..."), constructions with passer and liniaal, congruence and similarity
  proofs.
- Deciding which side is the hypotenuse or which angles are F-, Z- or overstaande hoeken.

## 6. Definition of done (per task)

- New acceptance cases in `tests/geometry/geometry_cases.yaml`, first failing, then passing on Hakan's machine.
- All geometry tests pass in geometry-lab (`py -m pytest -q`).
- A short section added to the end of this file: what was built, which cases cover it, open questions.
- Hakan commits in the local geometry-lab repo; nothing goes to GitHub until the module is finished.

## 6b. Integration (only when all tasks are done, by Hakan)

1. Copy `geometry/` and `tests/geometry/` into RAG-Demo. Do not copy `verify_math.py` or `pytest.ini`.
2. Run all RAG-Demo tests; both suites must pass together.
3. Connect `verify_math()` to `verify_geometry()` for the geometry check types (one small change).
4. One weekly PR as usual.

## 7. Background files (read before starting)

- `verify_math.py` — read the parser, `_rounded_compare` and the status/reason conventions.
- `tests/verify_math_cases.yaml` — the case format and the notation rules in `conventions`.
- `curriculum/vwo_onderbouw.yaml` and `curriculum/vwo_wiskunde_b_deel1.yaml` — topics T08, T28,
  T29, T30, T31, T39, B3.1, B3.5 are the ones this module covers.


## Progress Notes (27 Oct 2026)

### 1. Completed Tasks & Verification Scope
- **G1 (angle_sum):** Verified triangle, quadrilateral, straight, and full angles (`GEO001–GEO017`). Checks for degree symbols, rejects period decimals, and flags `triangle_sum_as_360` misconception.
- **G2 (shape_formula):** Perimeter, area, and volume calculations for 2D and 3D figures (`GEO018–GEO034`, `GEO097`, `GEO104–GEO105`, `GEO110`). Validates dimension units (`unit_mismatch`, `wrong_unit_dimension`, `wrong_unit`), respects exact $\pi$ forms, and handles `forgot_half_triangle` / `swapped_circumference_and_area`.
- **G3 (scale_factor):** Scaling for length ($k$), area ($k^2$), and volume ($k^3$) (`GEO035–GEO045`, `GEO101`, `GEO108`). Detects scaling misconceptions (`area_scaled_by_k`, `volume_scaled_by_k`, `volume_scaled_by_k2`), enforces $k > 0$, and requires identical original and answer units (`wrong_unit`).
- **G4 (trig_ratio):** Right-angled triangle ratios $\sin, \cos, \tan$ and inverse lookups (`GEO046–GEO057`, `GEO096`). Rejects radians (`radians_not_supported`), detects `swapped_sin_cos`, and skips false misconceptions at $45^\circ$.
- **G5 (sine_rule & cosine_rule):** Advanced triangle rules (`GEO058–GEO068`). Detects `ambiguous_triangle`, `impossible_triangle` (including obtuse angles where opposite side is not strictly longest), and `cosine_rule_added_instead_of_subtracted`. Scoped to `vwo4_wiskunde_b`.
- **G6 (point_distance & midpoint):** 2D coordinate calculations (`GEO069–GEO095`, `GEO098–GEO100`, `GEO102–GEO103`, `GEO106–GEO107`, `GEO109`). Fully supports Dutch semicolon separation for decimal commas (`A(1,5; 3)`), fraction coordinates (`1/2`), and safely catches malformed/code inputs without crashing or using `sp.sympify`. Flags `manhattan_distance`, `midpoint_subtracted_coordinates`, and `midpoint_forgot_divide_by_two`.

### 2. Contract & Quality Hardening (GEO096–GEO110)
- **Security:** Completely eliminated `sp.sympify` / `eval` on student inputs; parsing is delegated to `_safe_parse_expr` and `_numeric_value`.
- **Misconception Evaluation Order:** Ensured exact correctness is evaluated before student misconceptions across all check types.
- **Rounding:** Integrated standard `_rounded_compare` directly across all applicable check types, supporting both half-up decimal places and significant figures (`GEO100`, `GEO101`, `GEO107`). Unsupported instructions safely return `rounding_instruction_not_supported` (`GEO110`).
- **Output Standardization:** Standardized all return shapes via `_result(...)` helper ensuring presence of `status`, `check_type`, `input`, `submitted_answer`, `parser_invoked`, and teacher-facing `explanation_nl`.

### 3. Test Suite Status
- Total acceptance tests: 110 passed (0 failed).