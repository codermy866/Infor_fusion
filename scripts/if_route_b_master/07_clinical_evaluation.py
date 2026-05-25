#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def eval_sets() -> dict[str, pd.DataFrame]:
    out = {"HyDRA-DG source-only": C.load_source_preds(), "Best score-level TTA": C.best_tta_predictions()}
    clinical = C.read_csv(C.OUT / "predictions" / "clinical_baselines_predictions.csv")
    if clinical is not None:
        lr = clinical[clinical["method"].eq("Clinical logistic regression")].copy()
        if not lr.empty:
            out["Clinical logistic baseline"] = lr
    return out


def e40_dca() -> None:
    C.setup_plot_style()
    sets = eval_sets()
    rows = []
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2)
    for endpoint in ["pathology_cin2plus", "pathology_cin3plus"]:
        for t in thresholds:
            for name, df in sets.items():
                rows.append({"endpoint": endpoint, "threshold": t, "model": name, "net_benefit": C.net_benefit(df[endpoint], df["prob_cin2plus"], t)})
            y = sets["HyDRA-DG source-only"][endpoint].astype(int).to_numpy()
            prev = y.mean()
            rows.append({"endpoint": endpoint, "threshold": t, "model": "treat all", "net_benefit": prev - (1 - prev) * t / (1 - t)})
            rows.append({"endpoint": endpoint, "threshold": t, "model": "treat none", "net_benefit": 0.0})
    dca = pd.DataFrame(rows)
    dca.to_csv(C.OUT / "statistics" / "dca_results.csv", index=False, encoding="utf-8-sig")
    fig, axes = C.plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, endpoint in zip(axes, ["pathology_cin2plus", "pathology_cin3plus"]):
        sub = dca[dca["endpoint"].eq(endpoint)]
        for i, (name, g) in enumerate(sub.groupby("model")):
            lw = 1.8 if "HyDRA" in name or "TTA" in name else 1.0
            ax.plot(g["threshold"], g["net_benefit"], label=name, lw=lw, color=C.palette(8)[i])
        ax.set_title(endpoint.replace("pathology_", "").upper(), weight="bold")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.legend(fontsize=7, frameon=False)
    C.save_fig(fig, "Figure_DCA", "Decision-curve analysis compares source-only, best TTA, clinical baseline when available, treat-all, and treat-none strategies.")
    C.write_text(C.OUT / "paper_sections" / "sec_dca_summary.txt", "Decision-curve analysis was computed for CIN2+ and CIN3+ across thresholds from 0.05 to 0.50. Curves should be interpreted as clinical utility diagnostics under the locked LOCO protocol, not as deployment validation.\n")
    C.append_manifest("E40", "Decision Curve Analysis", "COMPLETED", ["figures/Figure_DCA.pdf", "statistics/dca_results.csv", "paper_sections/sec_dca_summary.txt"])


def calibration_bins(df: pd.DataFrame, endpoint: str, bins: int = 10):
    y = df[endpoint].astype(int).to_numpy()
    s = np.clip(df["prob_cin2plus"].astype(float).to_numpy(), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if m.sum() > 0:
            rows.append({"bin": i, "mean_pred": s[m].mean(), "observed": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def e41_calibration() -> None:
    C.setup_plot_style()
    sets = {k: v for k, v in eval_sets().items() if k in ["HyDRA-DG source-only", "Best score-level TTA"]}
    rows = []
    for name, df in sets.items():
        for centre, g in [("Pooled", df)] + list(df.groupby("held_out_center")):
            rows.append({"model": name, "centre": centre, "endpoint": "CIN2+", "ECE": C.ece_score(g["pathology_cin2plus"], g["prob_cin2plus"], 10), "n": len(g)})
            rows.append({"model": name, "centre": centre, "endpoint": "CIN3+", "ECE": C.ece_score(g["pathology_cin3plus"], g["prob_cin2plus"], 10), "n": len(g)})
    pd.DataFrame(rows).to_csv(C.OUT / "statistics" / "ece_by_centre.csv", index=False, encoding="utf-8-sig")
    fig, axes = C.plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, endpoint in zip(axes, ["pathology_cin2plus", "pathology_cin3plus"]):
        for i, (name, df) in enumerate(sets.items()):
            b = calibration_bins(df, endpoint)
            ax.plot(b["mean_pred"], b["observed"], marker="o", label=name, color=C.palette(4)[i], lw=2, markersize=4)
        ax.plot([0, 1], [0, 1], color=C.SCI_PALETTE[6], linestyle=":", lw=1.1)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.set_title(endpoint.replace("pathology_", "").upper(), weight="bold")
        ax.legend(fontsize=8, frameon=False)
    C.save_fig(fig, "Figure_Calibration_Curves", "Reliability diagrams compare source-only and best score-level TTA calibration under the locked LOCO evaluation.")
    C.write_text(C.OUT / "paper_sections" / "sec_calibration_summary.txt", "Calibration and ECE were computed for pooled data and each centre. These diagnostics show probability calibration behaviour separately from ranking metrics such as AUC.\n")
    C.append_manifest("E41", "Calibration and ECE", "COMPLETED", ["figures/Figure_Calibration_Curves.pdf", "statistics/ece_by_centre.csv", "paper_sections/sec_calibration_summary.txt"])


def e42_screening_efficiency() -> None:
    C.setup_plot_style()
    sets = {k: v for k, v in eval_sets().items() if k in ["HyDRA-DG source-only", "Best score-level TTA"]}
    rows = []
    for name, df in sets.items():
        y = df["pathology_cin3plus"].astype(int)
        s = df["prob_cin2plus"].astype(float)
        for t in np.linspace(0, 1, 101):
            pred = s >= t
            met = C.metrics_at_pred(y, s, pred.astype(int))
            rows.append({"model": name, "threshold": t, "CIN3+ sensitivity": met["sensitivity"], "CIN3+ NPV": met["npv"], "CIN3+ specificity": met["specificity"], "screen-positive rate": met["screen_positive_rate"], "FN": met["fn"]})
    curve = pd.DataFrame(rows)
    curve.to_csv(C.OUT / "statistics" / "screening_efficiency_curve.csv", index=False, encoding="utf-8-sig")
    fig, ax = C.plt.subplots(figsize=(7.5, 5.2))
    for i, (name, g) in enumerate(curve.groupby("model")):
        color = C.palette(4)[i]
        ax.plot(g["screen-positive rate"], g["CIN3+ sensitivity"], label=f"{name} sensitivity", color=color, lw=2)
        ax.plot(g["screen-positive rate"], g["CIN3+ NPV"], linestyle="--", label=f"{name} NPV", color=color, lw=1.6)
    ax.set_xlabel("Screen-positive rate")
    ax.set_ylabel("CIN3+ sensitivity / NPV")
    ax.set_title("Screening Efficiency", weight="bold")
    ax.legend(fontsize=7, frameon=False)
    C.save_fig(fig, "Figure_Screening_Efficiency", "Screen-positive rate versus CIN3+ sensitivity and NPV across thresholds.")
    C.write_text(C.OUT / "paper_sections" / "sec_screening_efficiency_summary.txt", "Screening-efficiency curves show the trade-off between CIN3+ sensitivity, NPV, and screen-positive rate across thresholds. The best TTA improved sensitivity at the cost of more screen-positive results.\n")
    C.append_manifest("E42", "Screening Efficiency", "COMPLETED", ["figures/Figure_Screening_Efficiency.pdf", "statistics/screening_efficiency_curve.csv", "paper_sections/sec_screening_efficiency_summary.txt"])


def main() -> None:
    C.ensure_dirs()
    e40_dca()
    e41_calibration()
    e42_screening_efficiency()


if __name__ == "__main__":
    main()
