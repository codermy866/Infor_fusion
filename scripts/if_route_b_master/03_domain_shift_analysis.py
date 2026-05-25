#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_PKG_OVERLAY = Path(__file__).resolve().parents[2] / ".routeb_python_pkgs"
if LOCAL_PKG_OVERLAY.exists() and str(LOCAL_PKG_OVERLAY) not in sys.path:
    # Domain-shift UMAP needs the scikit-learn version installed with umap-learn.
    sys.path.insert(0, str(LOCAL_PKG_OVERLAY))

try:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    PCA = None
    RandomForestClassifier = None
    confusion_matrix = None
    StratifiedKFold = None
    cross_val_predict = None
    cross_val_score = None
    StandardScaler = None

try:
    import umap
    UMAP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    umap = None
    UMAP_IMPORT_ERROR = repr(exc)

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def score_features() -> pd.DataFrame:
    lock = C.load_data_lock()[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
    src = C.load_source_preds()[["case_id", "prob_cin2plus"]].rename(columns={"prob_cin2plus": "score_dg_source"})
    hydra = C.read_csv(C.PATHS["main_hydra"])
    cols = ["case_id"]
    if hydra is not None:
        for c in ["prob_cin2plus", "alpha_colposcopy", "alpha_oct", "alpha_semantic", "uncertainty_colposcopy", "uncertainty_oct", "uncertainty_semantic", "delta_prior_to_semantic", "delta_semantic_to_colposcopy", "delta_colposcopy_to_oct"]:
            if c in hydra.columns:
                cols.append(c)
        hydra = hydra[cols].drop_duplicates("case_id").rename(columns={"prob_cin2plus": "score_hydra_full"})
    else:
        hydra = pd.DataFrame({"case_id": lock["case_id"]})
    out = lock.merge(src, on="case_id", how="left").merge(hydra, on="case_id", how="left")
    return out


def feature_sets() -> dict[str, tuple[pd.DataFrame, np.ndarray, list[str]]]:
    lock = C.load_data_lock()
    meta = lock[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
    clin = C.clinical_feature_frame(lock)
    sf = score_features()
    score_cols = [c for c in sf.columns if c not in ["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]]
    score = sf[score_cols].apply(pd.to_numeric, errors="coerce").fillna(sf[score_cols].median(numeric_only=True))
    combined = pd.concat([clin.reset_index(drop=True), score.reset_index(drop=True)], axis=1)
    return {
        "clinical-only": (meta, clin.to_numpy(float), list(clin.columns)),
        "score-only": (meta, score.to_numpy(float), score_cols),
        "combined": (meta, combined.to_numpy(float), list(combined.columns)),
    }


def e10_mmd() -> None:
    C.setup_plot_style()
    meta, x, source = C.load_locked_features()
    centres = list(meta["center_name"].dropna().astype(str).unique())
    mat = pd.DataFrame(index=centres, columns=centres, dtype=float)
    for ci in centres:
        xi = x[meta["center_name"].astype(str).eq(ci).to_numpy()]
        for cj in centres:
            xj = x[meta["center_name"].astype(str).eq(cj).to_numpy()]
            mat.loc[ci, cj] = 0.0 if ci == cj else C.rbf_mmd2(xi, xj)
    mat.to_csv(C.OUT / "statistics" / "mmd_matrix.csv", encoding="utf-8-sig")
    avg = pd.DataFrame({"Centre": centres, "Average outbound MMD": [mat.loc[c, [k for k in centres if k != c]].astype(float).mean() for c in centres]})
    avg = avg.sort_values("Average outbound MMD", ascending=False)
    avg.to_csv(C.OUT / "statistics" / "mmd_average_outbound.csv", index=False, encoding="utf-8-sig")
    label_mat = mat.rename(index=C.CENTRE_LABELS, columns=C.CENTRE_LABELS)
    plot_avg = avg.copy()
    plot_avg["Centre label"] = C.centre_label_series(plot_avg["Centre"])
    fig, axes = C.plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.1, 0.9]})
    if C.sns is not None:
        C.sns.heatmap(
            label_mat.astype(float),
            ax=axes[0],
            cmap=C.sns.blend_palette([C.SCI_PALETTE[7], C.SCI_PALETTE[1], C.SCI_PALETTE[4]], as_cmap=True),
            linewidths=0.6,
            linecolor="white",
            square=True,
            cbar_kws={"label": "MMD"},
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7},
        )
        C.sns.barplot(data=plot_avg, x="Average outbound MMD", y="Centre label", ax=axes[1], color=C.SCI_PALETTE[0])
    else:
        im = axes[0].imshow(label_mat.astype(float).values, cmap="viridis")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        axes[1].barh(plot_avg["Centre label"], plot_avg["Average outbound MMD"], color=C.SCI_PALETTE[0])
    axes[0].set_title("Pairwise feature-space MMD", weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Average outbound MMD")
    axes[1].set_ylabel("")
    axes[1].set_title("Average outbound shift", weight="bold")
    caption = f"MMD was computed in {source} space. The analysis describes available feature-space shift and does not by itself prove clinical failure or image-representation drift."
    C.save_fig(fig, "Figure_MMD_Matrix", caption)
    C.append_manifest("E10", "MMD Matrix and Heatmap", "COMPLETED_WITH_CAVEAT", ["statistics/mmd_matrix.csv", "statistics/mmd_average_outbound.csv", "figures/Figure_MMD_Matrix.pdf"], caption)


def nearest_centroid_cv(meta: pd.DataFrame, x: np.ndarray, names: list[str], set_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y = meta["center_name"].astype(str).to_numpy()
    labels = np.array(sorted(pd.unique(y)))
    x = np.nan_to_num(x, nan=np.nanmean(x, axis=0))
    idx_by_label = {lab: np.where(y == lab)[0] for lab in labels}
    rng = np.random.default_rng(C.SEED)
    fold_id = np.zeros(len(y), dtype=int)
    for lab, idx in idx_by_label.items():
        idx = idx.copy()
        rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, 5)):
            fold_id[part] = k
    preds = []
    accs = []
    for k in range(5):
        tr = fold_id != k
        te = ~tr
        xt, xv = C.standardize_train_test(x[tr], x[te])
        centroids = []
        for lab in labels:
            centroids.append(xt[y[tr] == lab].mean(axis=0))
        centroids = np.vstack(centroids)
        d = ((xv[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        pred = labels[np.argmin(d, axis=1)]
        preds.extend(zip(y[te], pred))
        accs.append(float((pred == y[te]).mean()))
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for true, pred in preds:
        cm.loc[true, pred] += 1
    overall = {"feature_set": set_name, "classifier": "nearest_centroid_cv_fallback", "mean_accuracy": np.mean(accs), "std_accuracy": np.std(accs), "n_features": x.shape[1]}
    # simple between-centre/within-centre variance ratio as feature importance
    x_std, _ = C.standardize_train_test(x, x)
    overall_mean = x_std.mean(axis=0)
    between = np.zeros(x.shape[1])
    within = np.zeros(x.shape[1])
    for lab in labels:
        part = x_std[y == lab]
        between += len(part) * (part.mean(axis=0) - overall_mean) ** 2
        within += ((part - part.mean(axis=0)) ** 2).sum(axis=0)
    ratio = between / np.maximum(within, 1e-8)
    imp = pd.DataFrame({"feature_set": set_name, "feature": names, "importance_ratio": ratio}).sort_values("importance_ratio", ascending=False).head(30)
    return overall, cm, imp


def random_forest_cv(meta: pd.DataFrame, x: np.ndarray, names: list[str], set_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y = meta["center_name"].astype(str).to_numpy()
    labels = np.array(sorted(pd.unique(y)))
    x = np.nan_to_num(x, nan=np.nanmean(x, axis=0))
    if StandardScaler is not None:
        x = StandardScaler().fit_transform(x)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=C.SEED, class_weight="balanced", n_jobs=-1)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
    pred = cross_val_predict(clf, x, y, cv=cv, n_jobs=-1)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[C.centre_label(v) for v in labels], columns=[C.centre_label(v) for v in labels])
    clf.fit(x, y)
    imp = pd.DataFrame({"feature_set": set_name, "feature": names, "importance_ratio": clf.feature_importances_}).sort_values("importance_ratio", ascending=False).head(30)
    overall = {"feature_set": set_name, "classifier": "RandomForestClassifier_5fold_cv", "mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std()), "n_features": x.shape[1]}
    return overall, cm, imp


def e11_centre_classifier() -> None:
    C.setup_plot_style()
    sets = feature_sets()
    results, imps = [], []
    cms = {}
    for name, (meta, x, cols) in sets.items():
        if RandomForestClassifier is not None:
            res, cm, imp = random_forest_cv(meta, x, cols, name)
        else:
            res, cm, imp = nearest_centroid_cv(meta, x, cols, name)
            cm = cm.rename(index=C.CENTRE_LABELS, columns=C.CENTRE_LABELS)
        results.append(res)
        imps.append(imp)
        cms[name] = cm
    res_df = pd.DataFrame(results)
    imp_df = pd.concat(imps, ignore_index=True)
    res_df.to_csv(C.OUT / "statistics" / "centre_classifier_results.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv(C.OUT / "statistics" / "centre_classifier_feature_importance.csv", index=False, encoding="utf-8-sig")
    fig, axes = C.plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (name, cm) in zip(axes, cms.items()):
        if C.sns is not None:
            C.sns.heatmap(
                cm,
                ax=ax,
                cmap=C.sns.blend_palette([C.SCI_PALETTE[7], C.SCI_PALETTE[1], C.SCI_PALETTE[0]], as_cmap=True),
                linewidths=0.5,
                linecolor="white",
                cbar=False,
                annot=True,
                fmt="d",
                annot_kws={"fontsize": 7},
            )
        else:
            ax.imshow(cm.values, cmap="Blues")
        ax.set_title(name.replace("-", " ").title(), weight="bold")
        ax.set_xlabel("Predicted centre")
        ax.set_ylabel("True centre")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    classifier_name = str(res_df["classifier"].iloc[0])
    caption = f"Centre separability was estimated with {classifier_name}. Combined feature accuracy was {res_df.loc[res_df['feature_set'].eq('combined'), 'mean_accuracy'].iloc[0]:.3f}."
    C.save_fig(fig, "Figure_Centre_Classifier_CM", caption)
    C.write_text(C.OUT / "paper_sections" / "sec_centre_classifier_summary.txt", caption + "\n")
    status = "COMPLETED" if RandomForestClassifier is not None else "COMPLETED_WITH_CAVEAT"
    note = "RandomForestClassifier used." if RandomForestClassifier is not None else "sklearn unavailable; nearest-centroid fallback used."
    C.append_manifest("E11", "Centre Classifier", status, ["statistics/centre_classifier_results.csv", "statistics/centre_classifier_feature_importance.csv", "figures/Figure_Centre_Classifier_CM.pdf"], note)


def e13_umap_pca() -> None:
    C.setup_plot_style()
    meta, x, source = C.load_locked_features()
    x_work = np.nan_to_num(x, nan=np.nanmean(x, axis=0))
    if StandardScaler is not None:
        x_work = StandardScaler().fit_transform(x_work)
    method = "PCA fallback"
    if umap is not None:
        if PCA is not None and x_work.shape[1] > 50:
            x_in = PCA(n_components=50, random_state=C.SEED).fit_transform(x_work)
        else:
            x_in = x_work
        emb = umap.UMAP(n_neighbors=30, min_dist=0.18, metric="euclidean", random_state=C.SEED).fit_transform(x_in)
        method = "UMAP"
    else:
        emb = C.pca_2d(x_work)
    plot = pd.DataFrame(
        {
            "case_id": meta["case_id"],
            "feature_source": source,
            "component_1": emb[:, 0],
            "component_2": emb[:, 1],
            "Centre": C.centre_label_series(meta["center_name"]),
            "CIN2+ label": np.where(meta["pathology_cin2plus"].astype(int).to_numpy() == 1, "CIN2+", "CIN2-"),
        }
    )
    plot.to_csv(C.OUT / "statistics" / "umap_input_feature_manifest.csv", index=False, encoding="utf-8-sig")
    centres = sorted(plot["Centre"].unique())
    fig, axes = C.plt.subplots(1, 2, figsize=(11.8, 4.8))
    if C.sns is not None:
        C.sns.scatterplot(
            data=plot,
            x="component_1",
            y="component_2",
            hue="Centre",
            hue_order=centres,
            palette=C.palette(len(centres)),
            s=20,
            alpha=0.72,
            linewidth=0,
            ax=axes[0],
        )
        C.sns.scatterplot(
            data=plot,
            x="component_1",
            y="component_2",
            hue="CIN2+ label",
            hue_order=["CIN2-", "CIN2+"],
            palette=[C.SCI_PALETTE[0], C.SCI_PALETTE[4]],
            s=20,
            alpha=0.64,
            linewidth=0,
            ax=axes[1],
        )
    else:
        for cen, color in zip(centres, C.palette(len(centres))):
            sub = plot[plot["Centre"].eq(cen)]
            axes[0].scatter(sub["component_1"], sub["component_2"], s=16, alpha=0.7, label=cen, color=color)
    axes[0].set_title("A. Centre distribution", weight="bold", loc="left")
    axes[1].set_title("B. CIN2+ label distribution", weight="bold", loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="best", title="")
    axes[1].legend(fontsize=8, frameon=False, loc="best", title="")
    for ax in axes:
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
    caption = f"{method} visualization in {source} space. Centre names are shown with English short labels for publication-safe font rendering."
    C.save_fig(fig, "Figure_UMAP_Centre_Distribution", caption)
    status = "COMPLETED" if umap is not None else "COMPLETED_WITH_CAVEAT"
    note = "UMAP used." if umap is not None else "PCA fallback because umap unavailable."
    C.append_manifest("E13", "UMAP Centre Distribution", status, ["figures/Figure_UMAP_Centre_Distribution.pdf", "statistics/umap_input_feature_manifest.csv"], note)


def binary_centre_separability(meta: pd.DataFrame, x: np.ndarray, centre: str, other: str) -> float:
    m = meta["center_name"].astype(str).isin([centre, other]).to_numpy()
    y = meta.loc[m, "center_name"].astype(str).eq(centre).to_numpy()
    xx = x[m]
    rng = np.random.default_rng(C.SEED)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.7 * len(idx))
    tr, te = idx[:split], idx[split:]
    xtr, xte = C.standardize_train_test(xx[tr], xx[te])
    c0 = xtr[y[tr] == 0].mean(axis=0)
    c1 = xtr[y[tr] == 1].mean(axis=0)
    d0 = ((xte - c0) ** 2).sum(axis=1)
    d1 = ((xte - c1) ** 2).sum(axis=1)
    pred = d1 < d0
    return float((pred == y[te]).mean()) if len(te) else np.nan


def e22_source_attribution() -> None:
    C.setup_plot_style()
    sets = feature_sets()
    hardest = "襄阳市中心医院"
    rows = []
    for set_name, (meta, x, cols) in sets.items():
        centres = [c for c in sorted(meta["center_name"].astype(str).unique()) if c != hardest]
        xi = x[meta["center_name"].astype(str).eq(hardest).to_numpy()]
        for centre in centres:
            src = x[meta["center_name"].astype(str).eq(centre).to_numpy()]
            rows.append(
                {
                    "target_centre": hardest,
                    "source_centre": centre,
                    "feature_set": set_name,
                    "mmd": C.rbf_mmd2(xi, src),
                    "binary_centre_separability": binary_centre_separability(meta, x, hardest, centre),
                    "interpretation": "feature-set comparison, not additive domain-shift decomposition",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "statistics" / "domain_shift_source_attribution.csv", index=False, encoding="utf-8-sig")
    plot_df = df.copy()
    plot_df["source_label"] = C.centre_label_series(plot_df["source_centre"])
    fig, ax = C.plt.subplots(figsize=(8.5, 4.6))
    if C.sns is not None:
        C.sns.barplot(data=plot_df, x="source_label", y="mmd", hue="feature_set", palette=C.palette(3), ax=ax)
    else:
        pivot = plot_df.pivot_table(index="source_label", columns="feature_set", values="mmd")
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MMD vs Xiangyang")
    ax.set_xlabel("Source centre")
    ax.set_title("Domain Shift Source Attribution", weight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, title="")
    C.save_fig(fig, "Figure_Domain_Shift_Source_Attribution", "Feature-set comparison of clinical-only, score-only, and combined discrepancies. This is source attribution rather than additive MMD decomposition.")
    C.write_text(C.OUT / "paper_sections" / "sec_domain_shift_source_attribution.txt", "Domain-shift source attribution compared clinical-only, score-only, and combined feature discrepancies against Xiangyang. These values are feature-set comparisons and should not be interpreted as additive contributions to a total shift.\n")
    C.append_manifest("E22", "Domain Shift Source Attribution", "COMPLETED_WITH_CAVEAT", ["statistics/domain_shift_source_attribution.csv", "figures/Figure_Domain_Shift_Source_Attribution.pdf"], "Source attribution, not additive decomposition.")


def main() -> None:
    C.ensure_dirs()
    e10_mmd()
    e11_centre_classifier()
    e13_umap_pca()
    e22_source_attribution()


if __name__ == "__main__":
    main()
