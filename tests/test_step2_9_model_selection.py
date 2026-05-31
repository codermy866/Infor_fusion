import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_top_model_configs_come_from_inner_selection():
    selection = pd.read_csv(OUT / "search/inner_centre_model_selection.csv")
    configs = json.loads((OUT / "search/top_dg_model_configs.json").read_text(encoding="utf-8"))
    selected = {x["model_variant"] for x in configs}
    assert selected
    assert selected.issubset(set(selection["model_variant"]))
    assert selection["safety_filter_pass"].dtype == bool or set(selection["safety_filter_pass"].astype(str)).issubset({"True", "False"})


def test_single_class_centre_metrics_are_na():
    centre = pd.read_csv(OUT / "statistics/centre_level_dg_metrics.csv")
    single = centre[centre["Notes"].fillna("").str.contains("single-class", regex=False)]
    assert not single.empty
    auc = single["AUC CIN2+"]
    assert (auc.isna() | auc.astype(str).str.upper().eq("NA")).all()
