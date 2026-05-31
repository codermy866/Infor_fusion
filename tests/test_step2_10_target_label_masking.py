from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"


def test_unlabelled_target_sets_omit_pathology_columns():
    target = pd.read_csv(OUT / "splits/unlabelled_target_sets.csv")
    assert len(target) == 1897
    assert target["case_id"].nunique() == 1897
    forbidden = {"pathology_cin2plus", "pathology_cin3plus", "label", "diagnosis"}
    assert forbidden.isdisjoint(set(target.columns))
    assert target["split_role"].eq("target_unlabelled").all()
    assert target["label_columns_removed"].astype(bool).all()


def test_label_masking_audit_exists():
    audit = OUT / "audit/target_label_masking_audit.md"
    assert audit.exists()
    text = audit.read_text(encoding="utf-8")
    assert "Pathology columns are deliberately omitted" in text
