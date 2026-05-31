from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_top_dg_predictions_cover_1897_per_seed():
    pred = pd.read_csv(OUT / "predictions/top_dg_model_loco_predictions.csv")
    counts = pred.groupby(["seed", "model_variant"])["case_id"].nunique()
    assert not counts.empty
    assert counts.min() == 1897


def test_dg_first_pass_outputs_have_expected_families():
    pred = pd.read_csv(OUT / "predictions/dg_first_pass_predictions.csv")
    assert pred["model_family"].eq("domain-generalisation adapter").all()
    assert pred["architecture"].eq("HyDRA-DG-Adapter").all()
    assert pred["case_id"].nunique() == 1897
