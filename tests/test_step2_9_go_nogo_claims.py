import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_go_nogo_status_and_package_exist():
    status = json.loads((OUT / "STEP2_9_DG_RECOVERY_STATUS.json").read_text(encoding="utf-8"))
    assert status["step2_9_status"] in {
        "PASSED_IF_ROUTE_A",
        "PASSED_IF_ROUTE_B_REFRAME",
        "PASSED_NO_IF_SUBMISSION_RECOMMENDED",
        "FAILED_PIPELINE",
    }
    assert status["if_recommendation"]["route"] in {"Route A", "Route B", "Route C"}
    for name in [
        "IF_GoNoGo_Report.md",
        "IF_Method_Reframe_If_Passed.md",
        "IF_Method_Reframe_If_Failed.md",
        "IF_Abstract_Options.md",
        "IF_Reviewer_Risk_Register.md",
    ]:
        assert (OUT / f"manuscript/{name}").exists()


def test_go_nogo_claims_block_full_hydra_if_not_active():
    status_text = (OUT / "STEP2_9_DG_RECOVERY_STATUS.md").read_text(encoding="utf-8")
    risk = (OUT / "manuscript/IF_Reviewer_Risk_Register.md").read_text(encoding="utf-8")
    assert "Full end-to-end HyDRA-CoE" in status_text
    assert "not active" in risk
    table5 = pd.read_csv(OUT / "tables/Table5_IF_Go_NoGo_Decision.csv")
    rebuild = table5[table5["Decision"].str.contains("Rebuild full end-to-end", regex=False)]
    assert not rebuild.empty
