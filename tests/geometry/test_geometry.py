from pathlib import Path
import sys
import yaml
import pytest

# Repository root dizinini sys.path'e ekle
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry import verify_geometry

CASES_FILE = Path(__file__).parent / "geometry_cases.yaml"


def load_cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])


CASES = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_geometry_case(case):
    kwargs = {
        "check_type": case["check_type"],
        "input": case.get("input", ""),
    }
    if "submitted_answer" in case:
        kwargs["submitted_answer"] = case["submitted_answer"]
    if "task_instruction" in case:
        kwargs["task_instruction"] = case["task_instruction"]
    if "values" in case:
        kwargs["values"] = case["values"]

    result = verify_geometry(**kwargs)

    expected = case["expected"]
    for key, val in expected.items():
        assert result.get(key) == val, (
            f"Fout bij case {case['id']} veld '{key}': "
            f"kreeg {result.get(key)!r}, verwachtte {val!r}"
        )