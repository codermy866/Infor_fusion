from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_6_active_full_runner_recovery"


def test_endpoint_reaudit_confirms_all_positive_centre_without_mapping_errors():
    centre = pd.read_csv(OUT / "audit/centre_endpoint_distribution_v2.csv")
    all_pos = centre[centre["n_pathology_cin2plus"] == centre["n_total"]]
    assert set(all_pos["center_name"]) == {"武大人民医院"}
    audit = pd.read_csv(OUT / "audit/all_positive_centre_endpoint_audit.csv")
    assert len(audit) == 89
    assert audit["pathology_cin2plus"].eq(1).all()


def test_raw_manifest_coverage_is_complete():
    manifest = pd.read_csv(OUT / "manifests/raw_image_manifest_n1897.csv")
    assert len(manifest) == 1897
    assert manifest["can_load_raw_oct"].mean() >= 0.95
    assert manifest["can_load_raw_colposcopy"].mean() >= 0.95
    assert manifest["can_load_trainable_adapter_features"].mean() >= 0.95

