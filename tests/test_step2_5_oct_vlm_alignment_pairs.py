from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def test_alignment_pairs_have_text_source():
    pairs = OUT / "oct_vlm_alignment/audit/oct_text_alignment_pairs.csv"
    assert pairs.exists()
    df = pd.read_csv(pairs)
    assert df["text"].notna().all()
    assert df["text_source"].notna().all()
    assert "diagnostic_label_used" in df.columns

