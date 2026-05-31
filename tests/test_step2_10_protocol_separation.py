from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"


def test_inductive_and_tta_tracks_are_reported_separately():
    table = pd.read_csv(OUT / "tables/Table1_Protocol_Separation.csv")
    assert set(table["Track"]) == {
        "Pure inductive LOCO",
        "Unlabelled target-centre adaptation / transductive TTA",
    }
    assert not table["Uses target labels"].astype(bool).any()
    source = table[table["Track"].eq("Pure inductive LOCO")].iloc[0]
    tta = table[table["Track"].str.contains("transductive TTA", regex=False)].iloc[0]
    assert not bool(source["Uses target-centre images/features"])
    assert bool(tta["Uses target-centre images/features"])


def test_protocol_report_names_forbidden_claims():
    text = (OUT / "audit/protocol_separation_report.md").read_text(encoding="utf-8")
    assert "Target-adapted performance" in text
    assert "label-tuned" in text or "label" in text
