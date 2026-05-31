from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_model_selection_uses_inner_validation_only():
    selection = pd.read_csv(OUT / "search/inner_centre_model_selection.csv")
    assert "mean_inner_validation_auc" in selection.columns
    assert not any(col.startswith("test_") or "outer_loco_auc" in col for col in selection.columns)

    weights = pd.read_csv(OUT / "ensembles/dg_ensemble_weights.csv")
    assert "selected_candidates" in weights.columns
    assert "test_auc" not in weights.columns


def test_prediction_split_roles_are_explicit():
    inner = pd.read_csv(OUT / "predictions/top_dg_model_loco_inner_validation_predictions.csv")
    test = pd.read_csv(OUT / "predictions/top_dg_model_loco_predictions.csv")
    assert set(inner["split_role"]) == {"inner_validation"}
    assert set(test["split_role"]) == {"test"}
