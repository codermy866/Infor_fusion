from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_domain_shift_metrics_exist():
    metrics = pd.read_csv(OUT / "audit/centre_shift_metrics.csv")
    required = {
        "Centre",
        "MMD vs pooled training centres",
        "CORAL covariance distance",
        "Feature centroid distance",
        "Centre classifier accuracy",
    }
    assert required.issubset(metrics.columns)
    assert len(metrics) >= 5
    assert metrics["Centre classifier accuracy"].between(0, 1).all()
    assert metrics["MMD vs pooled training centres"].notna().all()


def test_domain_shift_reports_exist():
    assert (OUT / "audit/centre_predictability_report.md").exists()
    assert (OUT / "audit/domain_shift_diagnostic_report.md").exists()
