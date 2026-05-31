from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publishable_v2" / "step2_main_loco"


def first_float(value):
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else None


def test_figures_have_source_csv():
    source = OUT / "figures" / "source"
    for name in ["Figure2_source.csv", "Figure3_source.csv", "Figure4_source.csv", "FigureS2_source.csv"]:
        assert (source / name).exists(), name


def test_figure_values_match_source_tables():
    table = pd.read_csv(OUT / "tables" / "Table2_Main_LOCO_Diagnostic_Performance.csv")
    fig = pd.read_csv(OUT / "figures" / "source" / "Figure2_source.csv")
    t_auc = first_float(table.loc[table["Method"].eq("HyDRA_CoE_Full"), "AUC (95% CI)"].iloc[0])
    f_auc = fig.loc[fig["model_name"].eq("HyDRA_CoE_Full"), "auc"].iloc[0]
    assert abs(t_auc - f_auc) < 0.002
