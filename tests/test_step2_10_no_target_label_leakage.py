from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"
CFG = ROOT / "configs/hydra_step2_10_target_adaptation_if_decision.yaml"


def test_tta_predictions_mark_no_target_label_use():
    pred = pd.read_csv(OUT / "predictions/tta_candidate_predictions.csv")
    assert not pred.empty
    assert pred["adaptation_track"].eq("transductive_tta").all()
    assert "used_target_labels" in pred.columns
    assert not pred["used_target_labels"].astype(bool).any()
    assert pred["used_target_features_without_labels"].astype(bool).all()


def test_thresholds_are_declared_source_inner_validation_only():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert cfg["thresholds"]["threshold_source"] == "source_inner_validation_only"
    inner = pd.read_csv(OUT / "predictions/tta_candidate_inner_validation_predictions.csv")
    assert set(inner["adaptation_track"]) == {"source_inner_validation_tta_simulation"}
    assert not inner["used_target_labels"].astype(bool).any()


def test_tta_training_report_blocks_label_tuning():
    text = (OUT / "audit/tta_training_report.md").read_text(encoding="utf-8")
    assert "Target labels were not used" in text
    assert "model selection" in text
    assert "marked unavailable" in text
