import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion"


def test_step2_8_status_and_required_outputs_exist():
    status = json.loads((OUT / "STEP2_8_AUC_RECOVERY_IFUSION_STATUS.json").read_text(encoding="utf-8"))
    assert status["step2_8_status"] in {
        "PASSED_IFUSION_AUC_RECOVERY",
        "PASSED_MODEST_AUC_RECOVERY",
        "PASSED_NO_AUC_GAIN_BUT_DIAGNOSTIC_COMPLETE",
        "FAILED_PIPELINE",
    }
    assert status["input_verification"]["status"] == "PASS"
    assert status["multiscan_dataset"]["n"] == 1897

    for name in [
        "Table1_AUC_Recovery_Experiment_Inventory",
        "Table2_Main_AUC_Recovery_Result",
        "Table3_Module_Contribution_AUC",
        "Table4_Centre_Wise_AUC_Safety",
        "Table5_Information_Fusion_Claim_Audit",
    ]:
        for ext in ["csv", "md", "tex"]:
            assert (OUT / f"tables/{name}.{ext}").exists()

    for name in [
        "Figure1_AUC_Recovery_Ladder",
        "Figure2_ROC_Comparison_Information_Fusion",
        "Figure3_Fusion_Contribution_Analysis",
        "Figure4_Centre_Generalisation_Domain_Shift",
        "Figure5_Safety_Constrained_Operating_Point",
    ]:
        for ext in ["pdf", "svg", "png"]:
            assert (OUT / f"figures/{name}.{ext}").exists()

    for name in [
        "IF_Method_Reframing.md",
        "IF_Results_Rewrite.md",
        "IF_Contribution_Statement.md",
        "IF_Limitations.md",
        "IF_Abstract_Update.md",
    ]:
        assert (OUT / f"manuscript/{name}").exists()


def test_step2_8_prediction_coverage_and_validation_only_selection():
    lock = pd.read_csv(ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv")
    locked_cases = set(lock["case_id"])
    ens = pd.read_csv(OUT / "predictions/auc_safety_ensemble_predictions.csv")
    assert set(ens["case_id"]) == locked_cases
    assert ens["case_id"].nunique() == 1897
    assert ens["split_role"].eq("test").all()
    assert not ens.duplicated(["case_id", "fold_id"]).any()

    top = pd.read_csv(OUT / "predictions/top_model_loco_predictions.csv")
    counts = top.groupby(["seed", "model_variant"])["case_id"].nunique()
    assert not counts.empty
    assert counts.min() == 1897
    assert set(top["split_role"]) == {"test"}

    search = pd.read_csv(OUT / "search/validation_search_results.csv")
    assert "validation_auc_cin2plus" in search.columns
    assert not any(col.startswith("test_") for col in search.columns)

    weights = pd.read_csv(OUT / "ensembles/ensemble_weights.csv")
    assert "selected_candidates" in weights.columns
    assert "validation_auc" not in weights.columns or "test_auc" not in weights.columns


def test_step2_8_metrics_and_claim_audit_are_honest():
    metrics = pd.read_csv(OUT / "statistics/auc_recovery_metrics.csv")
    methods = set(metrics["Method"])
    assert {
        "Step2 surrogate HyDRA",
        "Step2.6 active minimal adapter",
        "Best Step2.8 individual IFusion model",
        "Best Step2.8 AUC-safety ensemble",
    }.issubset(methods)

    ensemble = metrics[metrics["Method"] == "Best Step2.8 AUC-safety ensemble"].iloc[0]
    surrogate = metrics[metrics["Method"] == "Step2 surrogate HyDRA"].iloc[0]
    assert ensemble["AUC"] > surrogate["AUC"]
    assert ensemble["Safety eligible"] in [False, "False", 0]

    claim = pd.read_csv(OUT / "tables/Table5_Information_Fusion_Claim_Audit.csv")
    blocked = claim[claim["Claim"].str.contains("full end-to-end HyDRA-CoE", regex=False)].iloc[0]
    assert blocked["Supported?"] in [False, "False", 0]

    diag = (OUT / "audit/auc_recovery_diagnostic_report.md").read_text(encoding="utf-8")
    assert "corrected403" in diag
    assert "not directly comparable" in diag
