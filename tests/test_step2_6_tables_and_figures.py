from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery"


def test_full_method_not_claimed_when_full_runner_not_wired():
    table = pd.read_csv(OUT / "tables/Table_3_Active_HyDRA_Recovery.csv")
    full_rows = table[table["Method"].astype(str).str.contains("Active_Full", regex=False)]
    assert not full_rows.empty
    assert full_rows["Status"].astype(str).str.contains("NOT_RUN_FULL_END_TO_END_MODULES_NOT_WIRED").all()


def test_go_nogo_recommends_route_b_partial_recovery():
    table = pd.read_csv(OUT / "tables/Table_5_Go_NoGo_Recommendation.csv")
    row = table[table["Decision item"] == "Partial active HyDRA supported"].iloc[0]
    assert row["Pass/fail"] == "PASS"
    assert "trainable-adapter" in row["Recommended manuscript action"]


def test_step2_6_figures_and_sources_exist():
    for i in range(1, 6):
        assert (OUT / f"figures/source/Figure_{i}_source.csv").exists()
    for stem in [
        "Figure_1_Why_Step2_5_Failed_Partially",
        "Figure_2_Endpoint_Centre_Distribution",
        "Figure_3_Active_Runner_Recovery_Performance",
        "Figure_4_Auxiliary_OCT_SSL_VLM_Init_Effect",
        "Figure_5_Go_NoGo_Decision_Schematic",
    ]:
        for ext in ["pdf", "svg", "png"]:
            assert (OUT / "figures" / f"{stem}.{ext}").exists()

