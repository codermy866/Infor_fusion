#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "vlm02_feature_quality"
VLM01 = C.OUT / "vlm01_foldwise_lora"


FEATURE_SETS = {
    "clinical_only": {"groups": ["clinical", "text"], "frozen": False},
    "image_cached_adapter": {"groups": ["colpo", "oct"], "frozen": False},
    "combined_cached_adapter": {"groups": ["clinical", "text", "colpo", "oct"], "frozen": False},
    "frozen_cached_image": {"groups": ["colpo", "oct"], "frozen": True},
    "frozen_cached_combined": {"groups": ["clinical", "colpo", "oct"], "frozen": True},
}


def load_fold(fold_dir: Path):
    return C.read_table(fold_dir / "source_features.parquet"), C.read_table(fold_dir / "target_features.parquet")


def main() -> None:
    C.ensure_dirs()
    C.setup_plot_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fold_dirs = C.fold_dirs(VLM01)
    if not fold_dirs:
        raise SystemExit("No VLM01 fold feature directories found.")

    fold_rows = []
    pred_rows = []
    mmd_rows = []
    target_by_set: dict[str, list[pd.DataFrame]] = {k: [] for k in FEATURE_SETS}
    matrix_by_set: dict[str, tuple[np.ndarray, pd.Series]] = {}

    for fold_dir in fold_dirs:
        src, tgt = load_fold(fold_dir)
        fold_id = str(tgt["fold_id"].iloc[0])
        centre = str(tgt["centre"].iloc[0])
        centre_label = str(tgt["centre_label"].iloc[0])
        for fs, spec in FEATURE_SETS.items():
            metrics = C.linear_probe_fold(src, tgt, spec["groups"], frozen=spec["frozen"])
            score = metrics.pop("target_score")
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "held_out_centre": centre,
                    "held_out_centre_label": centre_label,
                    "feature_type": fs,
                    **{k: v for k, v in metrics.items() if not k.endswith("CI")},
                    "CIN2+ AUC 95% CI": metrics["CIN2+ AUC 95% CI"],
                }
            )
            for pid, y2, y3, s in zip(tgt["patient_id"], tgt["cin2_label"], tgt["cin3_label"], score):
                pred_rows.append(
                    {
                        "patient_id": pid,
                        "fold_id": fold_id,
                        "centre": centre,
                        "centre_label": centre_label,
                        "feature_type": fs,
                        "cin2_label": int(y2),
                        "cin3_label": int(y3),
                        "score": float(s),
                    }
                )
            x_src = C.feature_matrix(src, spec["groups"], frozen=spec["frozen"])
            x_tgt = C.feature_matrix(tgt, spec["groups"], frozen=spec["frozen"])
            mmd_rows.append(
                {
                    "fold_id": fold_id,
                    "source_vs_target": f"source_vs_{centre_label}",
                    "feature_type": fs,
                    "MMD": C.mmd_rbf(x_src, x_tgt),
                }
            )
            tmp = tgt[["patient_id", "centre", "centre_label", "cin2_label"]].copy()
            tmp["_matrix_index"] = np.arange(len(tmp))
            tmp["_feature_type"] = fs
            target_by_set[fs].append(tmp)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    C.write_csv(OUT / "vlm_feature_quality_by_fold.csv", fold_df)

    agg_rows = []
    for fs, g in pred_df.groupby("feature_type"):
        threshold = C.select_threshold_for_cin3(g["cin3_label"], g["score"])
        m = C.eval_binary_metrics(g["cin2_label"], g["cin3_label"], g["score"], threshold)
        agg_rows.append({"feature_type": fs, "n": len(g), "selected_global_eval_threshold": threshold, **m})
    agg_df = pd.DataFrame(agg_rows)
    C.write_csv(OUT / "vlm_feature_quality_metrics.csv", agg_df)
    C.write_csv(OUT / "vlm_linear_probe_predictions.csv", pred_df)

    mmd_df = pd.DataFrame(mmd_rows)
    C.write_csv(OUT / "vlm_source_target_mmd_by_fold.csv", mmd_df)

    # Pairwise centre MMD and centre classifier use one target copy per patient/fold.
    matrix_rows = []
    outbound_rows = []
    clf_rows = []
    umap_panels = []
    for fs, spec in FEATURE_SETS.items():
        all_df = []
        xs = []
        for fold_dir in fold_dirs:
            _, tgt = load_fold(fold_dir)
            all_df.append(tgt[["patient_id", "centre", "centre_label", "cin2_label"]].copy())
            xs.append(C.feature_matrix(tgt, spec["groups"], frozen=spec["frozen"]))
        meta = pd.concat(all_df, ignore_index=True)
        x = pad_vstack(xs)
        centres = list(meta["centre_label"].drop_duplicates())
        for a in centres:
            for b in centres:
                xa = x[meta["centre_label"].eq(a).to_numpy()]
                xb = x[meta["centre_label"].eq(b).to_numpy()]
                matrix_rows.append({"feature_type": fs, "centre_a": a, "centre_b": b, "MMD": C.mmd_rbf(xa, xb)})
        for a in centres:
            vals = [r["MMD"] for r in matrix_rows if r["feature_type"] == fs and r["centre_a"] == a and r["centre_b"] != a]
            outbound_rows.append({"feature_type": fs, "centre": a, "average_outbound_MMD": float(np.nanmean(vals))})

        y = meta["centre_label"].astype(str).to_numpy()
        if len(np.unique(y)) > 1:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=2026, stratify=y)
            clf = RandomForestClassifier(n_estimators=250, random_state=2026, class_weight="balanced_subsample")
            clf.fit(np.nan_to_num(x[tr]), y[tr])
            pred = clf.predict(np.nan_to_num(x[te]))
            acc = accuracy_score(y[te], pred)
        else:
            acc = float("nan")
        try:
            sil = silhouette_score(np.nan_to_num(x), meta["cin2_label"].astype(int), metric="euclidean")
        except Exception:
            sil = float("nan")
        clf_rows.append({"feature_type": fs, "centre_classifier_accuracy": acc, "cin2_silhouette": sil})

        reducer = PCA(n_components=2, random_state=2026)
        z = reducer.fit_transform(Standardize(np.nan_to_num(x)))
        panel = meta[["centre_label", "cin2_label"]].copy()
        panel["feature_type"] = fs
        panel["component_1"] = z[:, 0]
        panel["component_2"] = z[:, 1]
        umap_panels.append(panel)

    matrix_df = pd.DataFrame(matrix_rows)
    outbound_df = pd.DataFrame(outbound_rows)
    clf_df = pd.DataFrame(clf_rows)
    C.write_csv(OUT / "vlm_mmd_matrix_by_feature_type.csv", matrix_df)
    C.write_csv(OUT / "vlm_average_outbound_mmd.csv", outbound_df)
    C.write_csv(OUT / "vlm_centre_classifier_results.csv", clf_df)

    fig, ax = C.plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(data=outbound_df, x="feature_type", y="average_outbound_MMD", hue="centre", ax=ax, palette=C.SCI_PALETTE)
    ax.set_title("Source-Target Centre Shift by Feature Type", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Average outbound MMD")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    C.save_fig(fig, OUT / "figure_vlm_mmd_comparison")

    fig, ax = C.plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=agg_df, x="feature_type", y="CIN2+ AUC", ax=ax, palette=C.SCI_PALETTE)
    ax.axhline(0.741, color=C.SCI_PALETTE[4], linestyle="--", lw=1.2, label="Route B HyDRA-DG AUC")
    ax.set_title("Linear-Probe CIN2+ AUC", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Pooled CIN2+ AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    C.save_fig(fig, OUT / "figure_vlm_linear_probe_auc")

    panel_df = pd.concat(umap_panels, ignore_index=True)
    keep = ["clinical_only", "combined_cached_adapter", "frozen_cached_combined"]
    fig, axes = C.plt.subplots(1, len(keep), figsize=(13.5, 4.3))
    for ax, fs in zip(axes, keep):
        sub = panel_df[panel_df["feature_type"].eq(fs)]
        sns.scatterplot(
            data=sub,
            x="component_1",
            y="component_2",
            hue="centre_label",
            style="cin2_label",
            s=16,
            alpha=0.82,
            linewidth=0,
            ax=ax,
            palette=C.SCI_PALETTE,
        )
        ax.set_title(fs.replace("_", " "), weight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=7)
    C.save_fig(fig, OUT / "figure_vlm_umap_by_feature_type")

    claim_status = "NOT_TESTABLE"
    best_adapter = agg_df[agg_df["feature_type"].eq("combined_cached_adapter")]["CIN2+ AUC"].max()
    best_frozen = agg_df[agg_df["feature_type"].eq("frozen_cached_combined")]["CIN2+ AUC"].max()
    if pd.notna(best_adapter) and pd.notna(best_frozen):
        claim_status = "SUPPORTED_PARTIALLY" if best_adapter > best_frozen else "NOT_SUPPORTED_FOR_CACHED_ADAPTER"
    report = [
        "# VLM02 Feature-Quality Audit Report",
        "",
        f"Claim status for BioMedCLIP-LoRA/VLM-enhanced feature improvement: `{claim_status}`.",
        "",
        "BioMedCLIP-LoRA was not available; this audit compares cached feature adapters, frozen cached feature slices, and clinical-only features. Therefore BioMedCLIP-LoRA improvement remains not verified.",
        "",
        "## Aggregate Metrics",
        "",
        C.md_table(agg_df),
        "",
        "## Centre Classifier",
        "",
        C.md_table(clf_df),
    ]
    C.write_text(OUT / "vlm02_audit_report.md", "\n".join(report) + "\n")
    C.file_manifest(OUT, OUT / "vlm02_file_manifest.csv")


def Standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def pad_vstack(arrays: list[np.ndarray]) -> np.ndarray:
    width = max(a.shape[1] for a in arrays) if arrays else 0
    padded = []
    for a in arrays:
        if a.shape[1] < width:
            pad = np.zeros((a.shape[0], width - a.shape[1]), dtype=a.dtype)
            padded.append(np.hstack([a, pad]))
        else:
            padded.append(a)
    return np.vstack(padded)


if __name__ == "__main__":
    main()
