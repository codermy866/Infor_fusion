from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publishable_v2" / "step2_main_loco"
SPLIT = ROOT / "outputs" / "publishable_v2" / "splits" / "split_manifest_v2.csv"


def test_no_train_validation_test_leakage():
    manifest = pd.read_csv(SPLIT)
    for fold_id, fold in manifest.groupby("fold_id"):
        by_role = {
            role: set(group["patient_id"].astype(str))
            for role, group in fold.groupby("split_role")
        }
        assert by_role.get("train", set()).isdisjoint(by_role.get("validation", set())), fold_id
        assert by_role.get("train", set()).isdisjoint(by_role.get("test", set())), fold_id
        assert by_role.get("validation", set()).isdisjoint(by_role.get("test", set())), fold_id


def test_metric_undefined_single_class_handled_as_na():
    table = pd.read_csv(OUT / "tables" / "Table3_Centre_Wise_HyDRA_LOCO.csv")
    rows = table[(table["Endpoint"].eq("pathology_cin2plus")) & (table["CIN2+ positives, n"].eq(table["Test N"]))]
    assert not rows.empty
    assert rows["AUC (95% CI)"].astype(str).str.contains("NA").all()
    assert rows["Specificity (95% CI)"].astype(str).str.contains("NA").all()
