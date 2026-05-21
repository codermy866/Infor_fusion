#!/usr/bin/env python3
"""Publication-style figures from revision CSV tables (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parents[1]
PAPER_DIR = EXP_ROOT / "paper_revision"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_missing_modality(plt) -> None:
    path = TABLE_DIR / "missing_modality_robustness_metrics_formatted.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    focus = [
        "HyDRA_ELBO_StructuredPrior_AllCenter",
        "HyDRA_ELBO_AllCenter",
        "DirectLate_AllCenterPatientHoldout",
        "HyDRA_Full_Pretrained",
    ]
    sub = df[df["method"].astype(str).isin(focus)].copy()
    if sub.empty:
        return
    order = ["remove_colposcopy", "remove_oct", "remove_clinical_prior", "random_one", "random_two"]
    sub["setting"] = sub["setting"].astype(str)
    sub = sub[sub["setting"].isin(order)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    x = range(len(order))
    width = 0.18
    for i, method in enumerate(focus):
        msub = sub[sub["method"].eq(method)]
        if msub.empty:
            continue
        aucs = []
        for s in order:
            row = msub[msub["setting"].eq(s)]
            aucs.append(float(row["auc"].iloc[0]) if not row.empty else float("nan"))
        ax.bar([xi + (i - 1.5) * width for xi in x], aucs, width=width, label=method.replace("_", " "))
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_", " ") for s in order], rotation=18, ha="right")
    ax.set_ylabel("ROC-AUC (external stress evaluation)")
    ax.set_title("Missing-modality robustness (formatted table metrics)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fig_missing_modality_robustness.png")
    plt.close(fig)


def plot_input_corruption(plt) -> None:
    path = TABLE_DIR / "input_corruption_robustness_metrics_formatted.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    methods = [
        "HyDRA_ELBO_StructuredPrior_AllCenter",
        "HyDRA_ELBO_AllCenter",
        "Ablation_NoTrajectoryCoE",
    ]
    sub = df[df["method"].astype(str).isin(methods)].copy()
    if sub.empty:
        return
    piv = sub.pivot_table(index="corruption", columns="method", values="auc", aggfunc="first")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for col in piv.columns:
        ax.plot(range(len(piv)), piv[col].values, marker="o", linewidth=1.6, label=str(col).replace("_", " "))
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels(list(piv.index), rotation=22, ha="right")
    ax.set_ylabel("ROC-AUC under corruption")
    ax.set_title("Input corruption robustness (moderate severity, n=196 export)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_input_corruption_robustness.png")
    plt.close(fig)


def main() -> None:
    plt = _setup_matplotlib()
    plot_missing_modality(plt)
    plot_input_corruption(plt)
    print(f"Figures written under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
