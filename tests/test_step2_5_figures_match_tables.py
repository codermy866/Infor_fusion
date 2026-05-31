from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def _metric_value(x):
    import re

    s = str(x)
    if s.upper().startswith("NA"):
        return float("nan")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else float("nan")


def test_figures_have_source_csv():
    source = OUT / "figures/source"
    assert source.exists()
    for name in ["Figure_A_source.csv", "Figure_B_source.csv", "Figure_C_source.csv", "Figure_D_source.csv", "Figure_E_source.csv"]:
        assert (source / name).exists()
    for stem in [
        "Figure_A_Protocol_Implementation_Attribution",
        "Figure_B_Full_HyDRA_LOCO_Performance",
        "Figure_C_Full_Module_Ablation",
        "Figure_D_Auxiliary_OCT_VLM_Pretraining_Effect",
        "Figure_E_CIN3plus_Safety_Full_HyDRA",
    ]:
        for ext in ["pdf", "svg", "png"]:
            assert (OUT / "figures" / ("%s.%s" % (stem, ext))).exists()


def test_figure_values_match_tables():
    table_b = pd.read_csv(OUT / "tables/Table_B_Main_Full_HyDRA_LOCO_Performance.csv")
    fig_b = pd.read_csv(OUT / "figures/source/Figure_B_source.csv")
    table_auc = table_b["AUC (95% CI)"].map(_metric_value).fillna(-1).tolist()
    fig_auc = fig_b["auc_value"].fillna(-1).tolist()
    assert table_auc == fig_auc

