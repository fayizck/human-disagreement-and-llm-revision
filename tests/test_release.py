import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reference_integrity_reports_pass():
    study1 = json.loads(
        (ROOT / "study1/results/machine/integrity_checks.json").read_text()
    )
    study2 = json.loads(
        (ROOT / "study2/reconciliation/collection_reconciliation.json").read_text()
    )
    assert study1["status"] == "PASS"
    assert study2["passed"] is True


def test_reference_collection_accounting():
    study1 = json.loads(
        (
            ROOT
            / "study1/reconciliation/main_study_final_collection_reconciliation.json"
        ).read_text()
    )
    assert study1["status"] == "PASS"
    assert study1["checks"]["planned_slots_33000"]
    assert study1["checks"]["committed_32992"]
    assert study1["checks"]["structurally_ineligible_8"]

    study2 = json.loads(
        (ROOT / "study2/reconciliation/collection_reconciliation.json").read_text()
    )
    observed = (
        study2["planned_scientific_slots"],
        study2["committed_observations"],
        study2["structurally_ineligible"],
    )
    assert observed == (1_800, 1_776, 24)


def test_no_private_environment_files():
    names = {path.name for path in ROOT.rglob("*") if path.is_file()}
    assert ".env" not in names
