from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publishable_v2" / "step2_main_loco"


def test_thresholds_are_validation_derived():
    thresholds = pd.read_csv(OUT / "predictions" / "validation_thresholds_by_fold_model_seed.csv")
    assert thresholds["threshold_source"].eq("validation_only").all()
    assert thresholds[["threshold_youden", "threshold_safety95", "threshold_safety90"]].notna().all().all()


def test_no_legacy_985_loaded():
    status = (OUT / "STEP2_MAIN_LOCO_STATUS.json").read_text(encoding="utf-8")
    assert '"legacy_985_reference_detected_during_step2": false' in status
