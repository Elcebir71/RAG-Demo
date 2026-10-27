from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_math import verify_math  # noqa: E402

SUITE = yaml.safe_load((Path(__file__).parent / "verify_math_cases.yaml").read_text(encoding="utf-8"))


def build_input(case):
    if "input_generator" not in case:
        return case.get("input", "")
    gen = case["input_generator"]
    return gen["repeat"] * int(gen["times"]) + gen.get("suffix", "")


def run_case(case):
    check_type = case["check_type"]
    input_value = build_input(case)
    kwargs = {}
    # Newer fields are passed only when a case uses them, so older cases keep
    # working while the code for the new fields is still being written.
    for optional in ("variable", "substitutions"):
        if optional in case:
            kwargs[optional] = case[optional]
    return verify_math(
        check_type=check_type,
        input=input_value,
        submitted_answer=case.get("submitted_answer"),
        steps=case.get("steps"),
        task_instruction=case.get("task_instruction"),
        **kwargs,
    )


def _compare_expected(actual, expected, case_id):
    for key, value in expected.items():
        # YAML's `undefined` is a sentinel meaning no finite exact value should
        # be returned for division by zero; test the explicit reason instead.
        if key == "exact_value" and value == "undefined":
            assert actual.get("reason") == "division_by_zero", case_id
            continue
        assert actual.get(key) == value, f"{case_id}: field {key!r}: expected {value!r}, got {actual.get(key)!r}; full result={actual!r}"


@pytest.mark.parametrize("case", SUITE["cases"], ids=lambda case: case["id"])
def test_acceptance_case(case):
    result = run_case(case)
    _compare_expected(result, case["expected"], case["id"])


def test_unsafe_input_is_rejected_before_parser(monkeypatch):
    import verify_math as vm

    def parser_must_not_run(_text):
        raise AssertionError("parser must not be invoked for unsafe input")

    monkeypatch.setattr(vm, "_parse_expr", parser_must_not_run)
    result = vm.verify_math("arithmetic", "__import__('os')")
    assert result["status"] == "unverifiable"
    assert result["parser_invoked"] is False


def test_long_input_is_rejected_before_parser(monkeypatch):
    import verify_math as vm

    def parser_must_not_run(_text):
        raise AssertionError("parser must not be invoked for overlong input")

    monkeypatch.setattr(vm, "_parse_expr", parser_must_not_run)
    result = vm.verify_math("arithmetic", "1+" * 300 + "1")
    assert result["status"] == "unverifiable"
    assert result["reason"] == "input_too_long"
    assert result["parser_invoked"] is False


def test_negative_mixed_number_is_negative_whole_mixed_number():
    result = verify_math("arithmetic", "-1 3/4", "-7/4")
    assert result["status"] == "verified"
    assert result["exact_value"] == "-7/4"


def test_unsupported_decimal_rounding_instruction_is_unverifiable():
    # Significant figures are supported since 27 Oct 2026; rounding to tens is not.
    result = verify_math("equation_solution", "3x=1", "x=0,3", task_instruction="Rond af op tientallen.")
    assert result["status"] == "unverifiable"
    assert result["reason"] == "rounding_instruction_not_supported"


def test_expression_with_variable_is_not_accepted_as_arithmetic():
    result = verify_math("arithmetic", "2x + 1", "7")
    assert result["status"] == "unverifiable"
    assert result["reason"] == "not_numeric_expression"
