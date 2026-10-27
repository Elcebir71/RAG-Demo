"""Geometry verification module for Dutch secondary school curriculum (VWO onderbouw)."""

import re
from typing import Any
import sympy as sp

from verify_math import (
    _safe_parse_expr,
    _numeric_value,
    _parse_rounding_places,
    _rounded_compare,
    _serialize_exact,
    _MathInputError,
)

# Supported shapes and total angles for angle_sum
SUPPORTED_ANGLE_SHAPES: dict[str, dict[str, Any]] = {
    "driehoek": {"sum": 180, "expected_known": 2},
    "vierhoek": {"sum": 360, "expected_known": 3},
    "gestrekte hoek": {"sum": 180, "min_known": 1},
    "volle hoek": {"sum": 360, "min_known": 1},
}

# Unit dimensions mapping
UNIT_DIMENSIONS: dict[str, set[str]] = {
    "length": {"mm", "cm", "dm", "m", "km"},
    "area": {"mm²", "cm²", "dm²", "m²", "km²"},
    "volume": {
        "mm³",
        "cm³",
        "dm³",
        "m³",
        "km³",
        "l",
        "L",
        "ml",
        "mL",
        "cl",
        "cL",
        "dl",
        "dL",
    },
}

QUANTITY_DIMENSIONS: dict[str, str] = {
    "omtrek": "length",
    "oppervlakte": "area",
    "inhoud": "volume",
}

SCALE_FACTOR_POWERS: dict[str, int] = {
    "lengte": 1,
    "omtrek": 1,
    "oppervlakte": 2,
    "inhoud": 3,
}

SHAPE_MISCONCEPTION_NL: dict[str, str] = {
    "forgot_half_triangle": "De oppervlakte van een driehoek is ½ · basis · hoogte; de factor ½ is vergeten.",
    "swapped_circumference_and_area": "Omtrek en oppervlakte van de cirkel zijn verwisseld.",
}


def _result(
    status: str,
    check_type: str,
    input_str: str,
    submitted_answer: str | None = None,
    parser_invoked: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Construct standard response dictionary adhering to verify_math contract."""
    res = {
        "status": status,
        "check_type": check_type,
        "input": input_str,
        "submitted_answer": submitted_answer,
        "parser_invoked": parser_invoked,
    }
    res.update(kwargs)
    return res


def _has_dot_notation(text: str | None) -> bool:
    """Detects period notation in numeric context which is ambiguous in Dutch school notation."""
    if not text:
        return False
    return bool(re.search(r"\d\.\d|\d\.|\.\d", text))


def _has_radians(text: str | None) -> bool:
    """Detects radian specification in text."""
    if not text:
        return False
    return bool(
        re.search(
            r"\brad\b|\bradian\b|\bradiant\b|\bradiaal\b|\bradialen\b",
            text,
            re.IGNORECASE,
        )
    )


def _normalize_unit(unit_str: str | None) -> str | None:
    """Normalizes superscript digits in metric units."""
    if not unit_str:
        return None
    return unit_str.strip().replace("2", "²").replace("3", "³")


def _extract_unit_and_value(text: str) -> tuple[str, str | None]:
    """Extracts numeric expression string and trailing unit if present."""
    text = text.strip()
    unit_pattern = r"(mm[²2]|cm[²2]|dm[²2]|m[²2]|km[²2]|mm[³3]|cm[³3]|dm[³3]|m[³3]|km[³3]|km|dm|cm|mm|m|ml|mL|cl|cL|dl|dL|l|L)$"
    match = re.search(unit_pattern, text)
    if match:
        unit = _normalize_unit(match.group(1))
        val_str = text[: match.start()].strip()
        return val_str, unit
    return text, None


def _get_dimension(unit: str | None) -> str | None:
    """Returns the physical dimension category for a given unit."""
    if not unit:
        return None
    norm_unit = _normalize_unit(unit)
    for dim, units in UNIT_DIMENSIONS.items():
        if norm_unit in units:
            return dim
    return None


def _evaluate_shape_formula(
    quantity: str, shape: str, values: dict[str, Any]
) -> tuple[Any, dict[str, Any] | str]:
    """Computes exact formula expression and student misconceptions for shape_formula."""
    if shape == "rechthoek":
        l_val = values.get("l") if values.get("l") is not None else values.get("lengte")
        b_val = values.get("b") if values.get("b") is not None else values.get("breedte")
        if l_val is None or b_val is None:
            return None, "missing_variables"
        if quantity == "oppervlakte":
            return l_val * b_val, {}
        if quantity == "omtrek":
            return 2 * l_val + 2 * b_val, {}

    elif shape == "vierkant":
        z_val = values.get("z") if values.get("z") is not None else values.get("zijde")
        if z_val is None:
            return None, "missing_variables"
        if quantity == "oppervlakte":
            return z_val**2, {}
        if quantity == "omtrek":
            return 4 * z_val, {}

    elif shape == "driehoek":
        if quantity == "oppervlakte":
            b_val = values.get("basis") if values.get("basis") is not None else values.get("b")
            h_val = values.get("hoogte") if values.get("hoogte") is not None else values.get("h")
            if b_val is None or h_val is None:
                return None, "missing_variables"
            exact = sp.Rational(1, 2) * b_val * h_val
            misconceptions = {"forgot_half_triangle": b_val * h_val}
            return exact, misconceptions
        if quantity == "omtrek":
            sides = [
                values.get(k)
                for k in ["a", "b", "c", "zijde1", "zijde2", "zijde3"]
                if values.get(k) is not None
            ]
            if len(sides) == 3:
                return sum(sides), {}
            return None, "missing_variables"

    elif shape == "parallellogram":
        if quantity == "oppervlakte":
            b_val = values.get("basis") if values.get("basis") is not None else values.get("b")
            h_val = values.get("hoogte") if values.get("hoogte") is not None else values.get("h")
            if b_val is None or h_val is None:
                return None, "missing_variables"
            return b_val * h_val, {}
        if quantity == "omtrek":
            a_val = values.get("a")
            b_val = values.get("b")
            if a_val is None or b_val is None:
                return None, "missing_variables"
            return 2 * a_val + 2 * b_val, {}

    elif shape == "trapezium":
        if quantity == "oppervlakte":
            a_val = values.get("a")
            b_val = values.get("b")
            h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
            if a_val is None or b_val is None or h_val is None:
                return None, "missing_variables"
            return sp.Rational(1, 2) * (a_val + b_val) * h_val, {}

    elif shape == "cirkel":
        r_val = (
            values.get("r")
            if values.get("r") is not None
            else (
                values.get("straal")
                if values.get("straal") is not None
                else (values.get("d") / 2 if values.get("d") is not None else None)
            )
        )
        if r_val is None:
            return None, "missing_variables"
        circumference = 2 * sp.pi * r_val
        area = sp.pi * (r_val**2)
        if quantity == "omtrek":
            return circumference, {"swapped_circumference_and_area": area}
        if quantity == "oppervlakte":
            return area, {"swapped_circumference_and_area": circumference}

    elif shape == "balk":
        l_val = values.get("l") if values.get("l") is not None else values.get("lengte")
        b_val = values.get("b") if values.get("b") is not None else values.get("breedte")
        h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
        if l_val is None or b_val is None or h_val is None:
            return None, "missing_variables"
        if quantity == "inhoud":
            return l_val * b_val * h_val, {}
        if quantity == "oppervlakte":
            return 2 * (l_val * b_val + l_val * h_val + b_val * h_val), {}

    elif shape == "kubus":
        z_val = (
            values.get("z")
            if values.get("z") is not None
            else (
                values.get("ribbe")
                if values.get("ribbe") is not None
                else values.get("zijde")
            )
        )
        if z_val is None:
            return None, "missing_variables"
        if quantity == "inhoud":
            return z_val**3, {}
        if quantity == "oppervlakte":
            return 6 * (z_val**2), {}

    elif shape == "cilinder":
        r_val = values.get("r") if values.get("r") is not None else values.get("straal")
        h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
        if r_val is None or h_val is None:
            return None, "missing_variables"
        if quantity == "inhoud":
            return sp.pi * (r_val**2) * h_val, {}
        if quantity == "oppervlakte":
            return 2 * sp.pi * (r_val**2) + 2 * sp.pi * r_val * h_val, {}

    elif shape == "prisma":
        if quantity == "inhoud":
            g_val = values.get("G") if values.get("G") is not None else values.get("grondvlak")
            h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
            if g_val is None or h_val is None:
                return None, "missing_variables"
            return g_val * h_val, {}

    elif shape == "piramide":
        if quantity == "inhoud":
            g_val = values.get("G") if values.get("G") is not None else values.get("grondvlak")
            h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
            if g_val is None or h_val is None:
                return None, "missing_variables"
            return sp.Rational(1, 3) * g_val * h_val, {}

    elif shape == "kegel":
        r_val = values.get("r") if values.get("r") is not None else values.get("straal")
        if quantity == "inhoud":
            h_val = values.get("h") if values.get("h") is not None else values.get("hoogte")
            if r_val is None or h_val is None:
                return None, "missing_variables"
            return sp.Rational(1, 3) * sp.pi * (r_val**2) * h_val, {}
        if quantity == "oppervlakte":
            s_val = values.get("s")
            if r_val is None or s_val is None:
                return None, "missing_variables"
            return sp.pi * (r_val**2) + sp.pi * r_val * s_val, {}

    elif shape == "bol":
        r_val = values.get("r") if values.get("r") is not None else values.get("straal")
        if r_val is None:
            return None, "missing_variables"
        if quantity == "oppervlakte":
            return 4 * sp.pi * (r_val**2), {}
        if quantity == "inhoud":
            return sp.Rational(4, 3) * sp.pi * (r_val**3), {}

    return None, "unsupported_shape"


def _verify_angle_sum(
    input_str: str,
    submitted_answer: str | None,
) -> dict[str, Any]:
    """Handles check_type: angle_sum."""
    if _has_dot_notation(input_str) or _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    if ":" not in input_str:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="invalid_format", parser_invoked=False,
            explanation_nl="Invoer moet het formaat '<vorm>: <hoek>°, ...' hebben.",
        )

    shape_raw, angles_raw = input_str.split(":", 1)
    shape_name = shape_raw.strip().lower()
    angles_str = angles_raw.strip()

    if shape_name not in SUPPORTED_ANGLE_SHAPES:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="unsupported_shape", parser_invoked=False,
            explanation_nl=f"Geometrische vorm '{shape_name}' wordt niet ondersteund voor hoekensom.",
        )

    if not angles_str:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="not_enough_information", parser_invoked=False,
            explanation_nl="Geen hoeken opgegeven.",
        )

    parts = angles_str.split("°")
    if parts[-1].strip() != "":
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="ambiguous_comma_notation", parser_invoked=False,
            explanation_nl="Elke hoek in de invoer moet eindigen op een gradenteken (°).",
        )

    angle_tokens: list[str] = []
    for p in parts[:-1]:
        token = p.strip().lstrip(",").strip()
        if not token or not re.match(r"^-?\d+(?:,\d+)?$", token):
            return _result(
                "unverifiable", "angle_sum", input_str, submitted_answer,
                reason="ambiguous_comma_notation", parser_invoked=False,
                explanation_nl="Elke hoek in de invoer moet eindigen op een gradenteken (°).",
            )
        angle_tokens.append(token)

    shape_info = SUPPORTED_ANGLE_SHAPES[shape_name]
    expected_count = shape_info.get("expected_known", shape_info.get("min_known", 1))
    if len(angle_tokens) < expected_count:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="not_enough_information", parser_invoked=False,
            explanation_nl=f"Te weinig hoeken opgegeven voor {shape_name}.",
        )

    parsed_angles: list[Any] = []
    try:
        for token in angle_tokens:
            parsed_angles.append(_safe_parse_expr(token))
    except _MathInputError as e:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan de hoeken niet correct verwerken.",
        )

    total_sum = sp.Integer(shape_info["sum"])
    known_sum = sum(parsed_angles)
    remaining = total_sum - known_sum

    if any(a <= 0 for a in parsed_angles) or remaining <= 0:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="impossible_angles", parser_invoked=True,
            explanation_nl="De opgegeven hoeken zijn geometrisch onmogelijk.",
        )

    exact_value_str = _serialize_exact(remaining)

    if submitted_answer is None or submitted_answer.strip() == "":
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, unit="°", parser_invoked=True,
            explanation_nl=f"De ontbrekende hoek is {exact_value_str}°, maar er is geen antwoord ingeleverd.",
        )

    ans_clean = submitted_answer.strip().rstrip("°").strip()
    try:
        submitted_expr = _safe_parse_expr(ans_clean)
    except _MathInputError as e:
        return _result(
            "unverifiable", "angle_sum", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet correct verwerken.",
        )

    if bool(sp.simplify(submitted_expr - remaining) == 0):
        return _result(
            "verified", "angle_sum", input_str, submitted_answer,
            exact_value=exact_value_str, unit="°", parser_invoked=True,
            explanation_nl=f"Correct, de ontbrekende hoek is inderdaad {exact_value_str}°.",
        )

    res = _result(
        "wrong", "angle_sum", input_str, submitted_answer,
        exact_value=exact_value_str, unit="°", parser_invoked=True,
        explanation_nl=f"De ontbrekende hoek is {exact_value_str}°, niet {submitted_answer}.",
    )

    if shape_name == "driehoek":
        triangle_360_val = sp.Integer(360) - known_sum
        if sp.simplify(submitted_expr - triangle_360_val) == 0:
            res["misconception"] = "triangle_sum_as_360"
            res["explanation_nl"] = "De hoekensom van een driehoek is 180°, niet 360°."

    return res


def _verify_shape_formula(
    input_str: str,
    submitted_answer: str | None,
    task_instruction: str | None,
    values: dict[str, str] | None,
) -> dict[str, Any]:
    """Handles check_type: shape_formula."""
    if values:
        for val_str in values.values():
            if _has_dot_notation(val_str):
                return _result(
                    "unverifiable", "shape_formula", input_str, submitted_answer,
                    reason="ambiguous_dot_notation", parser_invoked=False,
                    explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
                )

    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    tokens = input_str.strip().lower().split(maxsplit=1)
    if len(tokens) != 2:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="invalid_format", parser_invoked=False,
            explanation_nl="Invoer moet het formaat '<grootheid> <vorm>' hebben.",
        )

    quantity, shape = tokens[0], tokens[1]
    if quantity not in QUANTITY_DIMENSIONS:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="unsupported_shape", parser_invoked=False,
            explanation_nl=f"Grootheid '{quantity}' wordt niet ondersteund.",
        )

    if not values:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Geen afmetingen opgegeven.",
        )

    val_units: list[str] = []
    parsed_values: dict[str, Any] = {}
    try:
        for key, raw_val in values.items():
            num_part, u = _extract_unit_and_value(raw_val)
            if u:
                val_units.append(u)
            parsed_values[key] = _safe_parse_expr(num_part)
    except _MathInputError as e:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan de afmetingen niet correct verwerken.",
        )

    if len(set(val_units)) > 1:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="unit_mismatch", parser_invoked=False,
            explanation_nl="Verschillende eenheden opgegeven in de afmetingen.",
        )

    formula_res, error_or_misc = _evaluate_shape_formula(quantity, shape, parsed_values)
    if formula_res is None:
        reason_code = str(error_or_misc)
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason=reason_code, parser_invoked=False,
            explanation_nl=f"Kan {quantity} van {shape} niet berekenen ({reason_code}).",
        )

    exact_value_str = _serialize_exact(formula_res)
    expected_dim = QUANTITY_DIMENSIONS[quantity]

    ans_num_str: str | None = None
    ans_unit: str | None = None
    if submitted_answer is not None and submitted_answer.strip() != "":
        ans_num_str, ans_unit = _extract_unit_and_value(submitted_answer)

    base_unit = val_units[0] if val_units else None
    expected_unit = None
    if base_unit:
        if expected_dim == "length":
            expected_unit = base_unit
        elif expected_dim == "area":
            expected_unit = f"{base_unit}²"
        elif expected_dim == "volume":
            expected_unit = f"{base_unit}³"

    if ans_unit:
        ans_dim = _get_dimension(ans_unit)
        if ans_dim != expected_dim:
            return _result(
                "wrong", "shape_formula", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="wrong_unit_dimension", parser_invoked=True,
                explanation_nl=f"Verkeerde eenheidsdimensie: verwacht een eenheid van {expected_dim}.",
            )
        if expected_unit and ans_unit != expected_unit:
            return _result(
                "wrong", "shape_formula", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="wrong_unit", parser_invoked=True,
                explanation_nl=f"Verkeerde eenheid: verwacht {expected_unit}.",
            )

    if submitted_answer is None or submitted_answer.strip() == "":
        out = _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"De exacte waarde is {exact_value_str}, maar er is geen antwoord ingeleverd.",
        )
        if ans_unit or expected_unit:
            out["unit"] = ans_unit or expected_unit
        return out

    try:
        submitted_expr = _safe_parse_expr(ans_num_str)
    except _MathInputError as e:
        return _result(
            "unverifiable", "shape_formula", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    effective_unit = ans_unit or expected_unit

    # 1. Exact match FIRST
    if bool(sp.simplify(submitted_expr - formula_res) == 0):
        out = _result(
            "verified", "shape_formula", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, het antwoord is {exact_value_str}.",
        )
        if effective_unit:
            out["unit"] = effective_unit
        return out

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(formula_res, submitted_expr, places, ans_num_str)
                if matched:
                    out = _result(
                        "verified", "shape_formula", input_str, submitted_answer,
                        exact_value=exact_value_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
                    if effective_unit:
                        out["unit"] = effective_unit
                    return out
        except _MathInputError as e:
            return _result(
                "unverifiable", "shape_formula", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    # 3. Check misconceptions ONLY when answer is not correct
    if isinstance(error_or_misc, dict):
        for misc_name, misc_val in error_or_misc.items():
            if bool(sp.simplify(misc_val - formula_res) == 0):
                continue
            if bool(sp.simplify(submitted_expr - misc_val) == 0):
                expl = SHAPE_MISCONCEPTION_NL.get(misc_name, f"Misconceptie: {misc_name}.")
                out = _result(
                    "wrong", "shape_formula", input_str, submitted_answer,
                    exact_value=exact_value_str, misconception=misc_name, parser_invoked=True,
                    explanation_nl=expl,
                )
                if effective_unit:
                    out["unit"] = effective_unit
                return out

    out = _result(
        "wrong", "shape_formula", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte waarde is {exact_value_str}.",
    )
    if effective_unit:
        out["unit"] = effective_unit
    return out


def _verify_scale_factor(
    input_str: str,
    submitted_answer: str | None,
    task_instruction: str | None,
    values: dict[str, str] | None,
) -> dict[str, Any]:
    """Handles check_type: scale_factor."""
    if values:
        for val_str in values.values():
            if _has_dot_notation(val_str):
                return _result(
                    "unverifiable", "scale_factor", input_str, submitted_answer,
                    reason="ambiguous_dot_notation", parser_invoked=False,
                    explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
                )

    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    quantity = input_str.strip().lower()
    if quantity not in SCALE_FACTOR_POWERS:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="unsupported_quantity", parser_invoked=False,
            explanation_nl=f"Grootheid '{quantity}' wordt niet ondersteund voor schaalfactor.",
        )

    if not values:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Geen waarden opgegeven voor schaalfactor.",
        )

    k_raw = values.get("k") if values.get("k") is not None else values.get("factor")
    orig_raw = (
        values.get("origineel")
        if values.get("origineel") is not None
        else (values.get("lengte") or values.get("oppervlakte") or values.get("inhoud") or values.get("waarde"))
    )

    if k_raw is None or orig_raw is None:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Zowel 'k' als 'origineel' moeten worden opgegeven.",
        )

    k_num_str, _ = _extract_unit_and_value(k_raw)
    orig_num_str, orig_unit = _extract_unit_and_value(orig_raw)

    try:
        k_expr = _safe_parse_expr(k_num_str)
        orig_expr = _safe_parse_expr(orig_num_str)
    except _MathInputError as e:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan de schaalfactor of originele waarde niet verwerken.",
        )

    if k_expr <= 0:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="impossible_scale_factor", parser_invoked=True,
            explanation_nl="Een vergrotingsfactor moet strikt positief zijn.",
        )

    power = SCALE_FACTOR_POWERS[quantity]
    exact_res = orig_expr * (k_expr**power)
    exact_value_str = _serialize_exact(exact_res)

    ans_num_str: str | None = None
    ans_unit: str | None = None
    if submitted_answer is not None and submitted_answer.strip() != "":
        ans_num_str, ans_unit = _extract_unit_and_value(submitted_answer)

    if orig_unit and ans_unit and ans_unit != orig_unit:
        return _result(
            "wrong", "scale_factor", input_str, submitted_answer,
            exact_value=exact_value_str, misconception="wrong_unit", parser_invoked=True,
            explanation_nl=f"Verkeerde eenheid: verwacht {orig_unit}.",
        )

    effective_unit = ans_unit or orig_unit

    if submitted_answer is None or submitted_answer.strip() == "":
        out = _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"De exacte geschaalde waarde is {exact_value_str}, maar er is geen antwoord ingeleverd.",
        )
        if effective_unit:
            out["unit"] = effective_unit
        return out

    try:
        submitted_expr = _safe_parse_expr(ans_num_str)
    except _MathInputError as e:
        return _result(
            "unverifiable", "scale_factor", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match FIRST
    if bool(sp.simplify(submitted_expr - exact_res) == 0):
        out = _result(
            "verified", "scale_factor", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, de geschaalde waarde is {exact_value_str}.",
        )
        if effective_unit:
            out["unit"] = effective_unit
        return out

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(exact_res, submitted_expr, places, ans_num_str)
                if matched:
                    out = _result(
                        "verified", "scale_factor", input_str, submitted_answer,
                        exact_value=exact_value_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
                    if effective_unit:
                        out["unit"] = effective_unit
                    return out
        except _MathInputError as e:
            return _result(
                "unverifiable", "scale_factor", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    # 3. Misconceptions check
    if quantity == "oppervlakte":
        scaled_k = orig_expr * k_expr
        if bool(sp.simplify(scaled_k - exact_res) != 0) and bool(sp.simplify(submitted_expr - scaled_k) == 0):
            out = _result(
                "wrong", "scale_factor", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="area_scaled_by_k", parser_invoked=True,
                explanation_nl="Bij een vergroting met factor k wordt de oppervlakte k² keer zo groot, niet k keer.",
            )
            if effective_unit:
                out["unit"] = effective_unit
            return out

    if quantity == "inhoud":
        scaled_k = orig_expr * k_expr
        if bool(sp.simplify(scaled_k - exact_res) != 0) and bool(sp.simplify(submitted_expr - scaled_k) == 0):
            out = _result(
                "wrong", "scale_factor", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="volume_scaled_by_k", parser_invoked=True,
                explanation_nl="Bij een vergroting met factor k wordt de inhoud k³ keer zo groot, niet k keer.",
            )
            if effective_unit:
                out["unit"] = effective_unit
            return out
        scaled_k2 = orig_expr * (k_expr**2)
        if bool(sp.simplify(scaled_k2 - exact_res) != 0) and bool(sp.simplify(submitted_expr - scaled_k2) == 0):
            out = _result(
                "wrong", "scale_factor", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="volume_scaled_by_k2", parser_invoked=True,
                explanation_nl="Bij een vergroting met factor k wordt de inhoud k³ keer zo groot, niet k² keer.",
            )
            if effective_unit:
                out["unit"] = effective_unit
            return out

    out = _result(
        "wrong", "scale_factor", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte waarde is {exact_value_str}.",
    )
    if effective_unit:
        out["unit"] = effective_unit
    return out


def _verify_trig_ratio(
    input_str: str,
    submitted_answer: str | None,
    task_instruction: str | None,
    values: dict[str, str] | None,
) -> dict[str, Any]:
    """Handles check_type: trig_ratio."""
    if values:
        for val_str in values.values():
            if _has_dot_notation(val_str):
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason="ambiguous_dot_notation", parser_invoked=False,
                    explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
                )
    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    if _has_radians(input_str):
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="radians_not_supported", parser_invoked=False,
            explanation_nl="Radialen worden niet ondersteund; gebruik graden (°).",
        )
    if values:
        for val_str in values.values():
            if _has_radians(val_str):
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason="radians_not_supported", parser_invoked=False,
                    explanation_nl="Radialen worden niet ondersteund; gebruik graden (°).",
                )

    match = re.match(
        r"^(sin|cos|tan)\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*=\s*(.+?)\s*/\s*(.+?)$",
        input_str.strip(),
        re.IGNORECASE,
    )
    if not match:
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="invalid_format", parser_invoked=False,
            explanation_nl="Invoer moet het formaat '<ratio>(<hoek>) = <teller> / <noemer>' hebben.",
        )

    ratio_name, angle_var, num_var, den_var = match.groups()
    ratio_name = ratio_name.lower()

    if not values:
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Geen waarden opgegeven.",
        )

    def get_var_value(name: str):
        if name in values:
            return values[name]
        for k, v in values.items():
            if k.strip().lower() == name.strip().lower():
                return v
        name_no_z = name.replace("zijde", "").strip().lower()
        for k, v in values.items():
            k_no_z = k.replace("zijde", "").strip().lower()
            if k_no_z == name_no_z:
                return v
        return None

    angle_raw = get_var_value(angle_var)
    num_raw = get_var_value(num_var)
    den_raw = get_var_value(den_var)

    knowns = [x for x in [angle_raw, num_raw, den_raw] if x is not None]
    if len(knowns) < 2:
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Onvoldoende gegevens om de ontbrekende waarde te berekenen.",
        )

    def eval_ratio(r_name: str | None, deg_expr):
        if not r_name:
            return None
        rad = deg_expr * sp.pi / 180
        if r_name == "sin":
            return sp.sin(rad)
        if r_name == "cos":
            return sp.cos(rad)
        if r_name == "tan":
            return sp.tan(rad)
        return None

    is_finding_angle = angle_raw is None
    target_unit = "°" if is_finding_angle else None

    exact_res = None
    misconception_res: dict[str, Any] = {}

    if is_finding_angle:
        num_clean, _ = _extract_unit_and_value(num_raw)
        den_clean, _ = _extract_unit_and_value(den_raw)
        try:
            num_expr = _safe_parse_expr(num_clean)
            den_expr = _safe_parse_expr(den_clean)
        except _MathInputError as e:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de zijdelengten niet verwerken.",
            )

        if num_expr <= 0 or den_expr <= 0:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason="impossible_trig_ratio", parser_invoked=True,
                explanation_nl="Zijdelengten moeten strikt positief zijn.",
            )

        if ratio_name in ("sin", "cos") and num_expr >= den_expr:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason="impossible_trig_ratio", parser_invoked=True,
                explanation_nl="In een rechthoekige driehoek kan de rechthoekszijde niet groter zijn dan de schuine zijde.",
            )

        frac = num_expr / den_expr
        if ratio_name == "sin":
            exact_res = sp.asin(frac) * 180 / sp.pi
        elif ratio_name == "cos":
            exact_res = sp.acos(frac) * 180 / sp.pi
        elif ratio_name == "tan":
            exact_res = sp.atan(frac) * 180 / sp.pi

    else:
        angle_clean = angle_raw.strip().rstrip("°").strip()
        try:
            deg_expr = _safe_parse_expr(angle_clean)
        except _MathInputError as e:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de hoekwaarde niet verwerken.",
            )

        if deg_expr <= 0 or deg_expr >= 90:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason="impossible_trig_ratio", parser_invoked=True,
                explanation_nl="De hoek in een rechthoekige driehoek moet tussen 0° en 90° liggen.",
            )

        ratio_val = eval_ratio(ratio_name, deg_expr)
        swapped_ratio_name = (
            "cos"
            if ratio_name == "sin"
            else ("sin" if ratio_name == "cos" else None)
        )
        swapped_ratio_val = eval_ratio(swapped_ratio_name, deg_expr)

        if num_raw is not None:
            num_clean, _ = _extract_unit_and_value(num_raw)
            try:
                num_expr = _safe_parse_expr(num_clean)
            except _MathInputError as e:
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                    explanation_nl="Kan de zijdelengte niet verwerken.",
                )
            if num_expr <= 0:
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason="impossible_trig_ratio", parser_invoked=True,
                    explanation_nl="Zijdelengten moeten strikt positief zijn.",
                )
            exact_res = num_expr / ratio_val
            if swapped_ratio_val:
                misconception_res["swapped_sin_cos"] = (
                    num_expr / swapped_ratio_val
                )

        elif den_raw is not None:
            den_clean, _ = _extract_unit_and_value(den_raw)
            try:
                den_expr = _safe_parse_expr(den_clean)
            except _MathInputError as e:
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                    explanation_nl="Kan de zijdelengte niet verwerken.",
                )
            if den_expr <= 0:
                return _result(
                    "unverifiable", "trig_ratio", input_str, submitted_answer,
                    reason="impossible_trig_ratio", parser_invoked=True,
                    explanation_nl="Zijdelengten moeten strikt positief zijn.",
                )
            exact_res = den_expr * ratio_val
            if swapped_ratio_val:
                misconception_res["swapped_sin_cos"] = (
                    den_expr * swapped_ratio_val
                )

    exact_res = sp.simplify(exact_res)
    exact_value_str = _serialize_exact(exact_res)

    if submitted_answer is None or submitted_answer.strip() == "":
        out = _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"De exacte waarde is {exact_value_str}, maar er is geen antwoord ingeleverd.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    ans_str = submitted_answer.strip()
    if "=" in ans_str:
        ans_str = ans_str.split("=")[-1].strip()

    ans_clean = ans_str.rstrip("°").strip()
    ans_clean, _ = _extract_unit_and_value(ans_clean)

    try:
        submitted_expr = _safe_parse_expr(ans_clean)
    except _MathInputError as e:
        return _result(
            "unverifiable", "trig_ratio", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match FIRST
    if bool(sp.simplify(submitted_expr - exact_res) == 0):
        out = _result(
            "verified", "trig_ratio", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, het antwoord is {exact_value_str}.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(exact_res, submitted_expr, places, ans_clean)
                if matched:
                    out = _result(
                        "verified", "trig_ratio", input_str, submitted_answer,
                        exact_value=exact_value_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
                    if target_unit:
                        out["unit"] = target_unit
                    return out
        except _MathInputError as e:
            return _result(
                "unverifiable", "trig_ratio", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    # 3. Misconceptions check (skip if misconception value equals exact value)
    for misc_name, misc_val in misconception_res.items():
        if bool(sp.simplify(misc_val - exact_res) == 0):
            continue
        if bool(sp.simplify(submitted_expr - misc_val) == 0):
            out = _result(
                "wrong", "trig_ratio", input_str, submitted_answer,
                exact_value=exact_value_str, misconception=misc_name, parser_invoked=True,
                explanation_nl="Sinus en cosinus zijn verwisseld.",
            )
            if target_unit:
                out["unit"] = target_unit
            return out

    out = _result(
        "wrong", "trig_ratio", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte waarde is {exact_value_str}.",
    )
    if target_unit:
        out["unit"] = target_unit
    return out


def _parse_sine_rule_input(text: str):
    """Parses 'a / sin(A) = b / sin(B)' into variable identifiers."""
    m = re.match(
        r"^\s*([a-zA-Z0-9_]+)\s*/\s*sin\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*=\s*([a-zA-Z0-9_]+)\s*/\s*sin\s*\(\s*([a-zA-Z0-9_]+)\s*\)\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _verify_sine_rule(
    input_str: str,
    submitted_answer: str | None,
    task_instruction: str | None,
    values: dict[str, str] | None,
) -> dict[str, Any]:
    """Handles check_type: sine_rule."""
    if values:
        for v in values.values():
            if _has_dot_notation(v):
                return _result(
                    "unverifiable", "sine_rule", input_str, submitted_answer,
                    reason="ambiguous_dot_notation", parser_invoked=False,
                    explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
                )
    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    if _has_radians(input_str) or (
        values and any(_has_radians(v) for v in values.values())
    ):
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="radians_not_supported", parser_invoked=False,
            explanation_nl="Radialen worden niet ondersteund; gebruik graden (°).",
        )

    parsed_input = _parse_sine_rule_input(input_str)
    if not parsed_input:
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="invalid_format", parser_invoked=False,
            explanation_nl="Invoer moet het formaat 'a / sin(A) = b / sin(B)' hebben.",
        )

    side_a_var, angle_a_var, side_b_var, angle_b_var = parsed_input

    if not values:
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Geen waarden opgegeven.",
        )

    def get_var_val(name: str):
        if not values:
            return None
        if name in values:
            return values[name]
        if f"hoek_{name}" in values:
            return values[f"hoek_{name}"]
        if f"hoek {name}" in values:
            return values[f"hoek {name}"]
        if f"zijde_{name}" in values:
            return values[f"zijde_{name}"]
        if f"zijde {name}" in values:
            return values[f"zijde {name}"]
        return None

    raw_side_a = get_var_val(side_a_var)
    raw_angle_a = get_var_val(angle_a_var)
    raw_side_b = get_var_val(side_b_var)
    raw_angle_b = get_var_val(angle_b_var)

    knowns = [
        x
        for x in [raw_side_a, raw_angle_a, raw_side_b, raw_angle_b]
        if x is not None
    ]
    if len(knowns) < 3:
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Onvoldoende gegevens voor de sinusregel.",
        )

    target_unit = None
    exact_res = None

    # Unknown is side
    if raw_side_b is None or raw_side_a is None:
        if raw_side_b is None:
            known_side_raw, known_opp_angle_raw, target_opp_angle_raw = (
                raw_side_a,
                raw_angle_a,
                raw_angle_b,
            )
        else:
            known_side_raw, known_opp_angle_raw, target_opp_angle_raw = (
                raw_side_b,
                raw_angle_b,
                raw_angle_a,
            )

        s_clean, _ = _extract_unit_and_value(known_side_raw)
        try:
            side_val = _safe_parse_expr(s_clean)
            a1_val = _safe_parse_expr(
                known_opp_angle_raw.strip().rstrip("°").strip()
            )
            a2_val = _safe_parse_expr(
                target_opp_angle_raw.strip().rstrip("°").strip()
            )
        except _MathInputError as e:
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de gegevens niet verwerken.",
            )

        if (
            side_val <= 0
            or a1_val <= 0
            or a2_val <= 0
            or (a1_val + a2_val >= 180)
        ):
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="De opgegeven hoeken of zijden vormen geen geldige driehoek.",
            )

        rad1 = a1_val * sp.pi / 180
        rad2 = a2_val * sp.pi / 180
        exact_res = sp.simplify(side_val * sp.sin(rad2) / sp.sin(rad1))

    # Unknown is angle
    else:
        if raw_angle_b is None:
            target_side_raw, opp_side_raw, opp_angle_raw = (
                raw_side_b,
                raw_side_a,
                raw_angle_a,
            )
        else:
            target_side_raw, opp_side_raw, opp_angle_raw = (
                raw_side_a,
                raw_side_b,
                raw_angle_b,
            )

        t_clean, _ = _extract_unit_and_value(target_side_raw)
        o_clean, _ = _extract_unit_and_value(opp_side_raw)
        a_clean = opp_angle_raw.strip().rstrip("°").strip()

        try:
            target_side = _safe_parse_expr(t_clean)
            opp_side = _safe_parse_expr(o_clean)
            opp_angle = _safe_parse_expr(a_clean)
        except _MathInputError as e:
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de gegevens niet verwerken.",
            )

        if (
            target_side <= 0
            or opp_side <= 0
            or opp_angle <= 0
            or opp_angle >= 180
        ):
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="De opgegeven waarden vormen geen geldige driehoek.",
            )

        opp_rad = opp_angle * sp.pi / 180
        sin_val = target_side * sp.sin(opp_rad) / opp_side

        sin_float = float(sin_val.evalf())
        if sin_float > 1.0000001 or sin_float <= 0:
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="Geen driehoek mogelijk: sin van de hoek is groter dan 1.",
            )

        # Obtuse angle with longer opposite side required
        if float(opp_angle) >= 90 and float(target_side) >= float(opp_side):
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="Geen driehoek mogelijk: bij een stompe hoek moet de overstaande zijde de langste zijn.",
            )

        # Ambiguous case: SSA with target_side > opp_side
        if float(target_side) > float(opp_side) and sin_float < 0.9999999:
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason="ambiguous_triangle", parser_invoked=True,
                explanation_nl="Dubbelzinnig geval van de sinusregel: er zijn twee mogelijke driehoeken.",
            )

        exact_res = sp.simplify(sp.asin(sin_val) * 180 / sp.pi)
        target_unit = "°"

    exact_value_str = _serialize_exact(exact_res)

    if submitted_answer is None or submitted_answer.strip() == "":
        out = _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"De exacte waarde is {exact_value_str}, maar er is geen antwoord ingeleverd.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    ans_str = submitted_answer.strip()
    if "=" in ans_str:
        ans_str = ans_str.split("=")[-1].strip()
    ans_clean = ans_str.rstrip("°").strip()
    ans_clean, _ = _extract_unit_and_value(ans_clean)

    try:
        submitted_expr = _safe_parse_expr(ans_clean)
    except _MathInputError as e:
        return _result(
            "unverifiable", "sine_rule", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match
    if bool(sp.simplify(submitted_expr - exact_res) == 0):
        out = _result(
            "verified", "sine_rule", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, het antwoord is {exact_value_str}.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(exact_res, submitted_expr, places, ans_clean)
                if matched:
                    out = _result(
                        "verified", "sine_rule", input_str, submitted_answer,
                        exact_value=exact_value_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
                    if target_unit:
                        out["unit"] = target_unit
                    return out
        except _MathInputError as e:
            return _result(
                "unverifiable", "sine_rule", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    out = _result(
        "wrong", "sine_rule", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte waarde is {exact_value_str}.",
    )
    if target_unit:
        out["unit"] = target_unit
    return out


def _parse_cosine_rule_input(text: str):
    """Parses 'a² = b² + c² - 2bc · cos(A)' into variables."""
    lhs_m = re.match(r"^\s*([a-zA-Z0-9_]+)(?:²|\^2)\s*=", text)
    if not lhs_m:
        return None
    lhs_side = lhs_m.group(1)

    cos_m = re.search(r"cos\s*\(\s*([a-zA-Z0-9_]+)\s*\)", text, re.IGNORECASE)
    if not cos_m:
        return None
    angle_var = cos_m.group(1)

    rhs_m = re.search(
        r"=\s*([a-zA-Z0-9_]+)(?:²|\^2)\s*\+\s*([a-zA-Z0-9_]+)(?:²|\^2)", text
    )
    if not rhs_m:
        return None
    b_var = rhs_m.group(1)
    c_var = rhs_m.group(2)

    return lhs_side, b_var, c_var, angle_var


def _verify_cosine_rule(
    input_str: str,
    submitted_answer: str | None,
    task_instruction: str | None,
    values: dict[str, str] | None,
) -> dict[str, Any]:
    """Handles check_type: cosine_rule."""
    if values:
        for v in values.values():
            if _has_dot_notation(v):
                return _result(
                    "unverifiable", "cosine_rule", input_str, submitted_answer,
                    reason="ambiguous_dot_notation", parser_invoked=False,
                    explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
                )
    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie in getallen is niet toegestaan; gebruik een komma als decimaalteken.",
        )

    if _has_radians(input_str) or (
        values and any(_has_radians(v) for v in values.values())
    ):
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="radians_not_supported", parser_invoked=False,
            explanation_nl="Radialen worden niet ondersteund; gebruik graden (°).",
        )

    parsed_input = _parse_cosine_rule_input(input_str)
    if not parsed_input:
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="invalid_format", parser_invoked=False,
            explanation_nl="Invoer moet het formaat 'a² = b² + c² - 2bc · cos(A)' hebben.",
        )

    lhs_side, b_var, c_var, angle_var = parsed_input

    if not values:
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Geen waarden opgegeven.",
        )

    def get_var_val(name: str):
        if not values:
            return None
        if name in values:
            return values[name]
        if f"hoek_{name}" in values:
            return values[f"hoek_{name}"]
        if f"hoek {name}" in values:
            return values[f"hoek {name}"]
        if f"zijde_{name}" in values:
            return values[f"zijde_{name}"]
        if f"zijde {name}" in values:
            return values[f"zijde {name}"]
        return None

    raw_a = get_var_val(lhs_side)
    raw_b = get_var_val(b_var)
    raw_c = get_var_val(c_var)
    raw_angle = get_var_val(angle_var)

    knowns = [x for x in [raw_a, raw_b, raw_c, raw_angle] if x is not None]
    if len(knowns) < 3:
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="missing_variables", parser_invoked=False,
            explanation_nl="Onvoldoende gegevens voor de cosinusregel.",
        )

    target_unit = None
    exact_res = None
    misconception_res: dict[str, Any] = {}

    # Unknown is side a
    if raw_a is None:
        b_clean, _ = _extract_unit_and_value(raw_b)
        c_clean, _ = _extract_unit_and_value(raw_c)
        ang_clean = raw_angle.strip().rstrip("°").strip()

        try:
            b_val = _safe_parse_expr(b_clean)
            c_val = _safe_parse_expr(c_clean)
            ang_val = _safe_parse_expr(ang_clean)
        except _MathInputError as e:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de gegevens niet verwerken.",
            )

        if b_val <= 0 or c_val <= 0 or ang_val <= 0 or ang_val >= 180:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="De opgegeven waarden vormen geen geldige driehoek.",
            )

        ang_rad = ang_val * sp.pi / 180
        a_sq = b_val**2 + c_val**2 - 2 * b_val * c_val * sp.cos(ang_rad)
        exact_res = sp.simplify(sp.sqrt(a_sq))

        # Check misconception: added instead of subtracted
        a_sq_misc = b_val**2 + c_val**2 + 2 * b_val * c_val * sp.cos(ang_rad)
        misconception_res["cosine_rule_added_instead_of_subtracted"] = (
            sp.simplify(sp.sqrt(a_sq_misc))
        )

    # Unknown is angle A
    else:
        a_clean, _ = _extract_unit_and_value(raw_a)
        b_clean, _ = _extract_unit_and_value(raw_b)
        c_clean, _ = _extract_unit_and_value(raw_c)

        try:
            a_val = _safe_parse_expr(a_clean)
            b_val = _safe_parse_expr(b_clean)
            c_val = _safe_parse_expr(c_clean)
        except _MathInputError as e:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
                explanation_nl="Kan de gegevens niet verwerken.",
            )

        if a_val <= 0 or b_val <= 0 or c_val <= 0:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="Zijdelengten moeten strikt positief zijn.",
            )

        # Triangle inequality check
        if (
            (a_val + b_val <= c_val)
            or (a_val + c_val <= b_val)
            or (b_val + c_val <= a_val)
        ):
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="De driehoeksongelijkheid geldt niet: geen driehoek mogelijk.",
            )

        cos_val = (b_val**2 + c_val**2 - a_val**2) / (2 * b_val * c_val)
        cos_float = float(cos_val.evalf())
        if abs(cos_float) > 1.0000001:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason="impossible_triangle", parser_invoked=True,
                explanation_nl="Geen driehoek mogelijk.",
            )

        exact_res = sp.simplify(sp.acos(cos_val) * 180 / sp.pi)
        target_unit = "°"

    exact_value_str = _serialize_exact(exact_res)

    if submitted_answer is None or submitted_answer.strip() == "":
        out = _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"De exacte waarde is {exact_value_str}, maar er is geen antwoord ingeleverd.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    ans_str = submitted_answer.strip()
    if "=" in ans_str:
        ans_str = ans_str.split("=")[-1].strip()
    ans_clean = ans_str.rstrip("°").strip()
    ans_clean, _ = _extract_unit_and_value(ans_clean)

    try:
        submitted_expr = _safe_parse_expr(ans_clean)
    except _MathInputError as e:
        return _result(
            "unverifiable", "cosine_rule", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_syntax"), parser_invoked=True,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match FIRST
    if bool(sp.simplify(submitted_expr - exact_res) == 0):
        out = _result(
            "verified", "cosine_rule", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, het antwoord is {exact_value_str}.",
        )
        if target_unit:
            out["unit"] = target_unit
        return out

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(exact_res, submitted_expr, places, ans_clean)
                if matched:
                    out = _result(
                        "verified", "cosine_rule", input_str, submitted_answer,
                        exact_value=exact_value_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
                    if target_unit:
                        out["unit"] = target_unit
                    return out
        except _MathInputError as e:
            return _result(
                "unverifiable", "cosine_rule", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    # 3. Misconceptions check
    for misc_name, misc_val in misconception_res.items():
        if bool(sp.simplify(misc_val - exact_res) == 0):
            continue
        if bool(sp.simplify(submitted_expr - misc_val) == 0):
            out = _result(
                "wrong", "cosine_rule", input_str, submitted_answer,
                exact_value=exact_value_str, misconception=misc_name, parser_invoked=True,
                explanation_nl="In de cosinusregel is 2bc·cos(A) opgeteld in plaats van afgetrokken.",
            )
            if target_unit:
                out["unit"] = target_unit
            return out

    out = _result(
        "wrong", "cosine_rule", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte waarde is {exact_value_str}.",
    )
    if target_unit:
        out["unit"] = target_unit
    return out


# =========================================================================
# G6 — Coordinates helpers and verifiers (point_distance & midpoint)
# =========================================================================

def _parse_coordinate_pair(coord_str: str) -> tuple[sp.Expr, sp.Expr]:
    """Parse a single 2D coordinate string like '(1, 2)' or '(2,5; 4)' using _numeric_value."""
    coord_str = coord_str.strip()
    if _has_dot_notation(coord_str):
        raise ValueError("ambiguous_dot_notation")

    m = re.search(r"\((.*?)\)", coord_str)
    if not m:
        raise ValueError("invalid_coordinate")
    inner = m.group(1).strip()

    # Case 1: Semicolon (;) used as coordinate separator (standard Dutch notation with decimals)
    if ";" in inner:
        parts = [p.strip() for p in inner.split(";")]
        if len(parts) != 2:
            raise ValueError("ambiguous_comma_notation")
        try:
            x_val = _numeric_value(parts[0])
            y_val = _numeric_value(parts[1])
            return x_val, y_val
        except Exception:
            raise ValueError("invalid_coordinate")

    # Case 2: Comma separator without semicolon
    commas_count = inner.count(",")
    if commas_count != 1:
        raise ValueError("ambiguous_comma_notation")

    if re.search(r"-?\d+,\d+", inner) and not re.search(r",\s+", inner):
        raise ValueError("ambiguous_comma_notation")

    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 2:
        raise ValueError("ambiguous_comma_notation")

    try:
        x_val = _numeric_value(parts[0])
        y_val = _numeric_value(parts[1])
        return x_val, y_val
    except Exception:
        raise ValueError("ambiguous_comma_notation")


def _extract_points(input_str: str) -> tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]:
    """Extract two points (x1, y1) and (x2, y2) from the input string."""
    if _has_dot_notation(input_str):
        raise ValueError("ambiguous_dot_notation")

    raw_points = re.findall(r"[A-Za-z]?\s*\([^)]*\)", input_str)
    if len(raw_points) != 2:
        if "," in input_str and ";" not in input_str:
            raise ValueError("ambiguous_comma_notation")
        raise ValueError("invalid_coordinate")

    pt1 = _parse_coordinate_pair(raw_points[0])
    pt2 = _parse_coordinate_pair(raw_points[1])
    return pt1, pt2


def _parse_submitted_point(ans_str: str) -> tuple[sp.Expr, sp.Expr]:
    """Safely parse submitted coordinate point like '(3, 4)', 'M(3, 4)' or '(2,5; 4)'."""
    clean = ans_str.strip()
    if "=" in clean:
        clean = clean.split("=")[-1].strip()
    clean = re.sub(r"^[A-Za-z]\s*", "", clean).strip()
    m = re.search(r"^\((.*?)\)$", clean)
    if not m:
        raise _MathInputError("invalid_format", "Geen geldige coördinatenstructuur.")
    inner = m.group(1).strip()

    if ";" in inner:
        parts = [p.strip() for p in inner.split(";")]
    elif "," in inner:
        parts = [p.strip() for p in inner.split(",")]
    else:
        raise _MathInputError("invalid_format", "Coördinaten moeten gescheiden zijn door een komma of puntkomma.")

    if len(parts) != 2:
        raise _MathInputError("invalid_format", "Verwacht 2 coördinaten.")

    x_expr = _safe_parse_expr(parts[0])
    y_expr = _safe_parse_expr(parts[1])
    return x_expr, y_expr


def _verify_point_distance(
    input_str: str,
    submitted_answer: str | None = None,
    task_instruction: str | None = None,
    values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify Euclidean distance between two points."""
    if not input_str and values and "A" in values and "B" in values:
        input_str = f"A{values['A']}, B{values['B']}"

    try:
        (x1, y1), (x2, y2) = _extract_points(input_str)
    except Exception as e:
        reason = str(e)
        if reason not in ("ambiguous_dot_notation", "ambiguous_comma_notation"):
            reason = "invalid_coordinate"
        return _result(
            "unverifiable", "point_distance", input_str, submitted_answer,
            reason=reason, parser_invoked=False,
            explanation_nl="Puntnotatie voor decimalen is niet toegestaan." if reason == "ambiguous_dot_notation" else "Coördinaten zijn ongeldig of dubbelzinnig genoteerd.",
        )

    dx = x2 - x1
    dy = y2 - y1
    exact_dist = sp.sqrt(dx**2 + dy**2)
    exact_value_str = _serialize_exact(exact_dist)
    exact_val_str = exact_value_str

    if submitted_answer is None or submitted_answer.strip() == "":
        return _result(
            "unverifiable", "point_distance", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_val_str, parser_invoked=True,
            explanation_nl="Geen antwoord opgegeven om te controleren.",
        )

    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "point_distance", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie voor decimalen is niet toegestaan.",
        )

    clean_sub = submitted_answer.strip()
    if "=" in clean_sub:
        clean_sub = clean_sub.split("=")[-1].strip()
    clean_sub, _ = _extract_unit_and_value(clean_sub)

    # Safe parse submitted answer without sp.sympify
    try:
        sub_expr = _safe_parse_expr(clean_sub)
    except Exception as e:
        return _result(
            "unverifiable", "point_distance", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_answer"), parser_invoked=False,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match FIRST
    if bool(sp.simplify(sub_expr - exact_dist) == 0):
        return _result(
            "verified", "point_distance", input_str, submitted_answer,
            exact_value=exact_val_str, parser_invoked=True,
            explanation_nl=f"Correct, de afstand is {exact_val_str}.",
        )

    # 2. Rounding check via _rounded_compare directly
    if task_instruction:
        try:
            places = _parse_rounding_places(task_instruction)
            if places is not None:
                matched, rounding_fields = _rounded_compare(exact_dist, sub_expr, places, clean_sub)
                if matched:
                    return _result(
                        "verified", "point_distance", input_str, submitted_answer,
                        exact_value=exact_val_str, parser_invoked=True,
                        explanation_nl="Correct afgerond.",
                        **rounding_fields,
                    )
        except _MathInputError as e:
            return _result(
                "unverifiable", "point_distance", input_str, submitted_answer,
                reason=e.reason, explanation_nl=e.message_nl,
            )

    # 3. Misconception: Manhattan distance
    manhattan_dist = abs(dx) + abs(dy)
    if sp.simplify(sub_expr - manhattan_dist) == 0 and sp.simplify(exact_dist - manhattan_dist) != 0:
        return _result(
            "wrong", "point_distance", input_str, submitted_answer,
            exact_value=exact_val_str, misconception="manhattan_distance", parser_invoked=True,
            explanation_nl="Afstand berekend als som van horizontale en verticale afstand in plaats van Pythagoras.",
        )

    return _result(
        "wrong", "point_distance", input_str, submitted_answer,
        exact_value=exact_val_str, parser_invoked=True,
        explanation_nl=f"Onjuist, de exacte afstand is {exact_val_str}.",
    )

def _verify_midpoint(
    input_str: str,
    submitted_answer: str | None = None,
    task_instruction: str | None = None,
    values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify midpoint coordinates M = ((x1 + x2)/2, (y1 + y2)/2)."""
    if not input_str and values and "A" in values and "B" in values:
        input_str = f"A{values['A']}, B{values['B']}"

    try:
        (x1, y1), (x2, y2) = _extract_points(input_str)
    except Exception as e:
        reason = str(e)
        if reason not in ("ambiguous_dot_notation", "ambiguous_comma_notation"):
            reason = "invalid_coordinate"
        return _result(
            "unverifiable", "midpoint", input_str, submitted_answer,
            reason=reason, parser_invoked=False,
            explanation_nl="Puntnotatie voor decimalen is niet toegestaan." if reason == "ambiguous_dot_notation" else "Coördinaten zijn ongeldig of dubbelzinnig genoteerd.",
        )

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    exact_value_str = f"({_serialize_exact(mx)}, {_serialize_exact(my)})"
    exact_val_str = exact_value_str

    if submitted_answer is None or submitted_answer.strip() == "":
        return _result(
            "unverifiable", "midpoint", input_str, submitted_answer,
            reason="no_claim_to_check", exact_value=exact_value_str, parser_invoked=True,
            explanation_nl="Geen antwoord opgegeven om te controleren.",
        )

    if _has_dot_notation(submitted_answer):
        return _result(
            "unverifiable", "midpoint", input_str, submitted_answer,
            reason="ambiguous_dot_notation", parser_invoked=False,
            explanation_nl="Puntnotatie voor decimalen is niet toegestaan.",
        )

    # Safe parse submitted point without sp.sympify
    try:
        sub_x, sub_y = _parse_submitted_point(submitted_answer)
    except Exception as e:
        return _result(
            "unverifiable", "midpoint", input_str, submitted_answer,
            reason=getattr(e, "reason", "invalid_answer"), parser_invoked=False,
            explanation_nl="Kan het ingeleverde antwoord niet verwerken.",
        )

    # 1. Exact match FIRST
    if sp.simplify(sub_x - mx) == 0 and sp.simplify(sub_y - my) == 0:
        return _result(
            "verified", "midpoint", input_str, submitted_answer,
            exact_value=exact_value_str, parser_invoked=True,
            explanation_nl=f"Correct, het midden is {exact_value_str}.",
        )

    # 2. Misconceptions
    subtracted_x = abs(x2 - x1) / 2
    subtracted_y = abs(y2 - y1) / 2
    if sp.simplify(sub_x - subtracted_x) == 0 and sp.simplify(sub_y - subtracted_y) == 0:
        if sp.simplify(sub_x - mx) != 0 or sp.simplify(sub_y - my) != 0:
            return _result(
                "wrong", "midpoint", input_str, submitted_answer,
                exact_value=exact_value_str, misconception="midpoint_subtracted_coordinates", parser_invoked=True,
                explanation_nl="Coördinaten zijn van elkaar afgetrokken in plaats van opgeteld.",
            )

    sum_x = x1 + x2
    sum_y = y1 + y2
    if sp.simplify(sub_x - sum_x) == 0 and sp.simplify(sub_y - sum_y) == 0:
        return _result(
            "wrong", "midpoint", input_str, submitted_answer,
            exact_value=exact_value_str, misconception="midpoint_forgot_divide_by_two", parser_invoked=True,
            explanation_nl="Vergeten te delen door 2 bij het berekenen van het midden.",
        )

    return _result(
        "wrong", "midpoint", input_str, submitted_answer,
        exact_value=exact_value_str, parser_invoked=True,
        explanation_nl=f"Onjuist, het exacte midden is {exact_value_str}.",
    )

def verify_geometry(
    check_type: str,
    input: str = "",
    submitted_answer: str | None = None,
    task_instruction: str | None = None,
    values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Public router for geometric claim verification."""
    if check_type == "angle_sum":
        return _verify_angle_sum(input, submitted_answer)
    if check_type == "shape_formula":
        return _verify_shape_formula(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "scale_factor":
        return _verify_scale_factor(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "trig_ratio":
        return _verify_trig_ratio(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "sine_rule":
        return _verify_sine_rule(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "cosine_rule":
        return _verify_cosine_rule(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "point_distance":
        return _verify_point_distance(
            input, submitted_answer, task_instruction, values
        )
    if check_type == "midpoint":
        return _verify_midpoint(
            input, submitted_answer, task_instruction, values
        )
    return _result(
        "unverifiable", check_type, input, submitted_answer,
        reason="unsupported_check_type", parser_invoked=False,
        explanation_nl=f"Controletype '{check_type}' wordt nog niet ondersteund.",
    )