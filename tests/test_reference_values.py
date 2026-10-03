"""The reference values that ports of the detector are tested against."""

import json
import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_reference_values.py"


def test_exports_cases_a_port_can_check(capsys):
    runpy.run_path(str(SCRIPT), run_name="__main__")
    data = json.loads(capsys.readouterr().out)
    assert data["generated_by"].startswith("blurfft ")
    assert data["model"]["samples"] > 100
    cases = data["cases"]
    assert [case["probability"] >= 0.5 for case in cases] == [False, True, False, False]
    for case in cases:
        assert len(case["pixels"]) == case["width"] * case["height"]
        assert all(isinstance(value, int) and 0 <= value <= 255 for value in case["pixels"])
        assert len(case["features"]) == 14
        assert case["radius"] > 0
    blur = data["blur"]
    assert len(blur["blurred"]) == len(blur["pixels"]) == blur["width"] * blur["height"]
