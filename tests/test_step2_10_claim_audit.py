import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"


def test_final_status_is_route_b_or_stricter_declared_state():
    status = json.loads((OUT / "STEP2_10_TARGET_ADAPTATION_IF_STATUS.json").read_text(encoding="utf-8"))
    assert status["step2_10_status"] in {
        "PASSED_IF_ROUTE_A_METHOD_PAPER",
        "PASSED_IF_ROUTE_B_DOMAIN_SHIFT_PAPER",
        "PASSED_ROUTE_C_REBUILD_REQUIRED",
        "FAILED_PIPELINE",
    }
    assert status["if_recommendation"]["route"] in {"Route A", "Route B", "Route C"}
    assert status["best_method"]["uses_target_labels"] is False


def test_claim_audit_blocks_unimplemented_full_hydra_claims():
    table = pd.read_csv(OUT / "tables/Table5_Final_Claim_Audit.csv")
    blocked = table[table["Supported?"].astype(str).isin(["False", "false", "0"])]
    text = " ".join(blocked["Claim"].astype(str).tolist())
    assert "full end-to-end HyDRA-CoE" in text
    assert "supervised CoE trajectory learning" in text
    assert "Hua_Xi/XiangYa labelled external validation" in text


def test_manuscript_decision_package_exists():
    for name in [
        "FINAL_IF_DECISION_REPORT.md",
        "IF_Route_A_Method_Paper_Draft_If_Passed.md",
        "IF_Route_B_Domain_Shift_Benchmark_Draft.md",
        "IF_Route_C_Rebuild_Full_Runner_Plan.md",
        "Reviewer_Risk_Register.md",
    ]:
        assert (OUT / f"manuscript/{name}").exists()
