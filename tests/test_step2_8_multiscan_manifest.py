import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion"


def test_step2_8_multiscan_manifest_locked_n1897_and_columns():
    lock = pd.read_csv(ROOT / "outputs/publishable_v2/data_lock/data_lock_n1897.csv")
    manifest = pd.read_csv(OUT / "manifests/multiscan_multiview_manifest.csv")
    assert len(manifest) == 1897
    assert set(manifest["case_id"]) == set(lock["case_id"])

    required = {
        "case_id",
        "center_name",
        "pathology_cin2plus",
        "pathology_cin3plus",
        "oct_selected_bscans_5_json",
        "oct_selected_bscans_10_json",
        "oct_selected_bscans_20_json",
        "colposcopy_selected_images_json",
        "oct_quality_summary",
        "colposcopy_quality_summary",
        "clinical_prior_available",
        "vlm_cache_available",
    }
    assert required.issubset(manifest.columns)
    assert manifest["clinical_prior_available"].all()
    assert manifest["vlm_cache_available"].all()


def test_step2_8_multiscan_manifest_uses_multiple_views_deterministically():
    manifest = pd.read_csv(OUT / "manifests/multiscan_multiview_manifest.csv")
    oct5 = manifest["oct_selected_bscans_5_json"].map(json.loads)
    oct10 = manifest["oct_selected_bscans_10_json"].map(json.loads)
    oct20 = manifest["oct_selected_bscans_20_json"].map(json.loads)
    col = manifest["colposcopy_selected_images_json"].map(json.loads)

    assert oct5.map(len).min() >= 1
    assert oct5.map(len).max() <= 5
    assert oct10.map(len).max() <= 10
    assert oct20.map(len).max() <= 20
    assert col.map(len).min() >= 1
    assert col.map(len).max() > 1

    # Deterministic subsets should be stable and unique within each case.
    for paths in oct20.head(100):
        assert len(paths) == len(set(paths))

    feature_table = pd.read_csv(OUT / "manifests/multiscan_feature_table.csv")
    assert len(feature_table) == 1897
    assert feature_table["case_id"].nunique() == 1897
