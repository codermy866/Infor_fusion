from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def test_no_auxiliary_diagnostic_label_leakage():
    pairs = OUT / "oct_vlm_alignment/audit/oct_text_alignment_pairs.csv"
    df = pd.read_csv(pairs)
    aux = df[df["is_auxiliary_oct"].astype(str).str.lower() == "true"]
    assert not aux.empty
    assert not aux["uses_pathology_label"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
    assert not aux["diagnostic_label_used"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
    banned = ["cin2", "cin3", "cancer", "benign", "high-grade", "high grade"]
    text = " ".join(aux["text"].astype(str).str.lower().tolist())
    assert not any(token in text for token in banned)

