from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"


EXPECTED_METHODS = {
    "target centre normalisation",
    "source free coral",
    "source free mmd",
    "prototype distribution alignment no labels",
    "confidence filtered pseudo label no threshold from test labels",
}


def test_tta_candidates_cover_locked_loco_cases_per_method():
    pred = pd.read_csv(OUT / "predictions/tta_candidate_predictions.csv")
    assert set(pred["method"]) == EXPECTED_METHODS
    counts = pred.groupby("method")["case_id"].nunique()
    assert len(counts) == len(EXPECTED_METHODS)
    assert counts.min() == 1897
    assert pred["prob_cin2plus"].between(0, 1).all()


def test_source_only_reference_is_inductive_loco():
    source = pd.read_csv(OUT / "predictions/source_only_reference_predictions.csv")
    assert source["case_id"].nunique() == 1897
    assert set(source["adaptation_track"]) == {"inductive_loco"}
    assert not source["used_target_labels"].astype(bool).any()


def test_tta_metrics_include_source_and_best_tta_candidate():
    metrics = pd.read_csv(OUT / "statistics/tta_metrics.csv")
    assert "Step2.9 source-only DG ensemble" in set(metrics["Method"])
    assert EXPECTED_METHODS.issubset(set(metrics["Method"]))
    assert metrics["AUC"].between(0, 1).all()
    table2 = pd.read_csv(OUT / "tables/Table2_Main_Target_Adaptation_Result.csv")
    assert "Best TTA candidate" in set(table2["Method"])
