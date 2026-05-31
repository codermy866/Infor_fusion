from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def test_predictions_cover_1897_cases_per_seed_variant():
    status_path = OUT / "audit/full_hydra_variant_run_status.csv"
    assert status_path.exists()
    status = pd.read_csv(status_path)
    completed = status[status["status"].astype(str).str.startswith("DONE")]
    for _, row in completed.iterrows():
        pred = ROOT / row["prediction_path"]
        assert pred.exists()
        df = pd.read_csv(pred)
        for seed, group in df.groupby("seed"):
            assert group["case_id"].nunique() == 1897


def test_thresholds_are_validation_only():
    step2 = ROOT / "outputs/publishable_v2/step2_main_loco/predictions/validation_thresholds_by_fold_model_seed.csv"
    assert step2.exists()
    df = pd.read_csv(step2)
    assert {"threshold_safety95", "threshold_safety90", "threshold_youden"}.issubset(df.columns)


def test_single_class_centre_metrics_are_na():
    table = ROOT / "outputs/publishable_v2/step2_main_loco/tables/Table3_Centre_Wise_HyDRA_LOCO.csv"
    df = pd.read_csv(table)
    row = df[(df["Held-out centre"] == "武大人民医院") & (df["Endpoint"] == "pathology_cin2plus")].iloc[0]
    assert str(row["AUC (95% CI)"]).startswith("NA")
    assert str(row["Specificity (95% CI)"]).startswith("NA")

