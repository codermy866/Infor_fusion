from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery"


def test_dataloader_and_overfit_gates_pass():
    smoke = pd.read_csv(OUT / "audit/dataloader_smoke_test_results.csv")
    overfit = pd.read_csv(OUT / "audit/overfit_sanity_results.csv")
    assert smoke["passed"].all()
    assert overfit["passed"].all()


def test_completed_active_predictions_cover_1897_cases():
    for path in [
        OUT / "predictions/active_visual_baseline_predictions.csv",
        OUT / "predictions/active_hydra_minimal_predictions.csv",
    ]:
        pred = pd.read_csv(path)
        for (model, seed), group in pred.groupby(["model_variant", "seed"]):
            assert group["case_id"].nunique() == 1897
            assert set(group["split_role"]) == {"test"}


def test_thresholds_are_fold_specific_validation_outputs():
    for path in [
        OUT / "predictions/active_visual_baseline_predictions_thresholds.csv",
        OUT / "predictions/active_hydra_minimal_predictions_thresholds.csv",
    ]:
        thr = pd.read_csv(path)
        assert {"threshold_safety95", "threshold_safety90", "threshold_youden", "validation_auc"}.issubset(thr.columns)
        assert thr["fold_id"].nunique() == 5

