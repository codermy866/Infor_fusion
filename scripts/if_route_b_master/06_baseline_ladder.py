#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    StandardScaler = None

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def metric_rows(pred: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    rows = []
    for method, g in pred.groupby(method_col):
        score = g["prob_cin2plus"].astype(float)
        pred_col = "pred_t_cin3_safety95" if "pred_t_cin3_safety95" in g.columns else ("pred_t_safety95" if "pred_t_safety95" in g.columns else None)
        if pred_col:
            yhat = g[pred_col].astype(float) >= 0.5
        else:
            yhat = score >= 0.5
        cin2 = C.metrics_at_pred(g["pathology_cin2plus"], score, yhat.astype(int))
        cin3 = C.metrics_at_pred(g["pathology_cin3plus"], score, yhat.astype(int))
        centre_col = "held_out_center" if "held_out_center" in g.columns else "center_name"
        rows.append(
            {
                "Method": method,
                "CIN2+ AUC": cin2["auc"],
                "CIN2+ AP": cin2["average_precision"],
                "CIN3+ AUC": cin3["auc"],
                "CIN3+ sensitivity": cin3["sensitivity"],
                "CIN3+ FN": cin3["fn"],
                "Screen-positive rate": cin2["screen_positive_rate"],
                "Centre gap": C.centre_gap(g.rename(columns={centre_col: "held_out_center"}), "held_out_center") if "prob_cin2plus" in g else np.nan,
                "Comparability status": "LOCKED_LOCO_COMPARABLE",
            }
        )
    return pd.DataFrame(rows)


def e30_clinical_baselines() -> None:
    df = C.load_data_lock()
    clin = C.clinical_feature_frame(df)
    centres = sorted(df["center_name"].astype(str).unique())
    rows = []
    for centre in centres:
        train = ~df["center_name"].astype(str).eq(centre)
        test = ~train
        g = df[test].copy()
        feat = clin[test]
        hpv_score = np.where(feat["hpv16_18_positive"] > 0, 0.85, np.where(feat["other_hr_hpv"] > 0, 0.65, 0.15))
        tct_score = np.where(feat["tct_high_grade"] > 0, 0.90, np.where(feat["tct_abnormal"] > 0, 0.70, 0.20))
        for method, score in [("HPV-only rule", hpv_score), ("TCT-only rule", tct_score)]:
            rec = g[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
            rec["held_out_center"] = centre
            rec["method"] = method
            rec["prob_cin2plus"] = score
            rec["pred_t_cin3_safety95"] = (score >= 0.5).astype(int)
            rows.append(rec)
        xtr, xte = C.standardize_train_test(clin[train].to_numpy(float), clin[test].to_numpy(float))
        w = C.fit_logistic_gd(xtr, df.loc[train, "pathology_cin2plus"].to_numpy(int))
        score = C.predict_logistic(xte, w)
        train_score = C.predict_logistic(xtr, w)
        th = C.threshold_for_sensitivity(df.loc[train, "pathology_cin3plus"].to_numpy(int), train_score, 0.95)
        rec = g[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
        rec["held_out_center"] = centre
        rec["method"] = "Clinical logistic regression"
        rec["prob_cin2plus"] = score
        rec["threshold_source_cin3_safety95"] = th
        rec["pred_t_cin3_safety95"] = (score >= th).astype(int)
        rows.append(rec)
        if RandomForestClassifier is not None:
            scaler = StandardScaler()
            xtr_rf = scaler.fit_transform(clin[train].to_numpy(float))
            xte_rf = scaler.transform(clin[test].to_numpy(float))
            rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=12, class_weight="balanced", random_state=C.SEED, n_jobs=-1)
            rf.fit(xtr_rf, df.loc[train, "pathology_cin2plus"].to_numpy(int))
            score_rf = rf.predict_proba(xte_rf)[:, 1]
            train_score_rf = rf.predict_proba(xtr_rf)[:, 1]
            th_rf = C.threshold_for_sensitivity(df.loc[train, "pathology_cin3plus"].to_numpy(int), train_score_rf, 0.95)
            rec = g[["case_id", "center_name", "pathology_cin2plus", "pathology_cin3plus"]].copy()
            rec["held_out_center"] = centre
            rec["method"] = "Clinical random forest"
            rec["prob_cin2plus"] = score_rf
            rec["threshold_source_cin3_safety95"] = th_rf
            rec["pred_t_cin3_safety95"] = (score_rf >= th_rf).astype(int)
            rows.append(rec)
    pred = pd.concat(rows, ignore_index=True)
    C.write_pred(pred, "clinical_baselines_predictions.csv")
    met = metric_rows(pred)
    if RandomForestClassifier is None:
        met = pd.concat(
            [
                met,
                pd.DataFrame(
                    [
                        {
                            "Method": "Clinical random forest",
                            "CIN2+ AUC": "NA",
                            "CIN2+ AP": "NA",
                            "CIN3+ AUC": "NA",
                            "CIN3+ sensitivity": "NA",
                            "CIN3+ FN": "NA",
                            "Screen-positive rate": "NA",
                            "Centre gap": "NA",
                            "Comparability status": "BLOCKED_NO_SKLEARN",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    met.to_csv(C.OUT / "statistics" / "clinical_baselines_metrics.csv", index=False, encoding="utf-8-sig")
    C.write_table(met, "Table_Clinical_Baselines_IF")
    status = "COMPLETED" if RandomForestClassifier is not None else "COMPLETED_WITH_CAVEAT"
    note = "Clinical random forest included." if RandomForestClassifier is not None else "Random forest blocked because sklearn unavailable."
    C.append_manifest("E30", "Clinical Baselines", status, ["predictions/clinical_baselines_predictions.csv", "statistics/clinical_baselines_metrics.csv", "tables/Table_Clinical_Baselines_IF.csv"], note)


def aggregate_all_models() -> pd.DataFrame:
    allm = C.read_csv(C.PATHS["main_all_models"])
    if allm is None:
        return pd.DataFrame()
    group = ["model_name", "case_id"]
    keep = ["center_name", "held_out_center", "pathology_cin2plus", "pathology_cin3plus"]
    rows = []
    for (model, case), g in allm.groupby(group, sort=False):
        row = {k: g[k].iloc[0] for k in keep if k in g.columns}
        row["method"] = model
        row["case_id"] = case
        row["prob_cin2plus"] = pd.to_numeric(g["prob_cin2plus"], errors="coerce").mean()
        if "pred_t_safety95" in g:
            row["pred_t_cin3_safety95"] = int(pd.to_numeric(g["pred_t_safety95"], errors="coerce").mean() >= 0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def e31_e32_e33_existing_baselines() -> None:
    agg = aggregate_all_models()
    if agg.empty:
        for eid, name in [("E31", "Unimodal Baselines"), ("E32", "Simple Fusion Baselines"), ("E33", "Extended Ablation")]:
            C.append_manifest(eid, name, "BLOCKED", [], "All-model predictions missing.")
        return
    uni = agg[agg["method"].isin(["ColposcopyOnly_ViT", "OCTOnly_ViT"])].copy()
    if not uni.empty:
        C.write_pred(uni, "unimodal_baselines_predictions.csv")
        met = metric_rows(uni)
        met.to_csv(C.OUT / "statistics" / "unimodal_baselines_metrics.csv", index=False, encoding="utf-8-sig")
        audit = met[["Method", "Comparability status"]].copy()
        audit["Audit note"] = "Existing locked all-model predictions; no new checkpoint training performed."
        audit.to_csv(C.OUT / "tables" / "Table_Unimodal_Baselines_Audit.csv", index=False, encoding="utf-8-sig")
        C.append_manifest("E31", "Unimodal Baselines", "COMPLETED_WITH_CAVEAT", ["predictions/unimodal_baselines_predictions.csv", "statistics/unimodal_baselines_metrics.csv", "tables/Table_Unimodal_Baselines_Audit.csv"], "Existing predictions only.")
    fusion_models = ["ColposcopyOCT_EarlyConcat", "ColposcopyOCT_LateFusion", "ColposcopyOCTText_CrossAttention", "HyDRA_CoE_Full"]
    fus = agg[agg["method"].isin(fusion_models)].copy()
    if not fus.empty:
        C.write_pred(fus, "simple_fusion_baselines_predictions.csv")
        met = metric_rows(fus)
        met.to_csv(C.OUT / "statistics" / "simple_fusion_baselines_metrics.csv", index=False, encoding="utf-8-sig")
        C.write_table(met, "Table_Simple_Fusion_Baselines_IF")
        C.append_manifest("E32", "Simple Fusion Baselines", "COMPLETED_WITH_CAVEAT", ["predictions/simple_fusion_baselines_predictions.csv", "statistics/simple_fusion_baselines_metrics.csv", "tables/Table_Simple_Fusion_Baselines_IF.csv"], "Existing all-model predictions, not newly trained.")
    abl = metric_rows(agg)
    abl["Comparability status"] = "LOCKED_LOCO_COMPARABLE_EXISTING_OUTPUT"
    abl.to_csv(C.OUT / "statistics" / "ablation_extended_metrics.csv", index=False, encoding="utf-8-sig")
    C.write_table(abl, "Table_Ablation_Extended_IF")
    C.append_manifest("E33", "Extended HyDRA Ablation", "COMPLETED_WITH_CAVEAT", ["statistics/ablation_extended_metrics.csv", "tables/Table_Ablation_Extended_IF.csv"], "Existing all-model predictions used.")


def final_ladder() -> None:
    skel = C.read_csv(C.OUT / "tables" / "Table_Fusion_Ladder_IF_skeleton.csv")
    if skel is None:
        skel = pd.DataFrame()
    rows = []
    if not skel.empty:
        rows.extend(skel.to_dict("records"))
    for path_s, label in [
        (C.OUT / "statistics" / "unimodal_baselines_metrics.csv", "existing unimodal"),
        (C.OUT / "statistics" / "simple_fusion_baselines_metrics.csv", "existing fusion"),
    ]:
        met = C.read_csv(path_s)
        if met is not None:
            for _, r in met.iterrows():
                rows.append(
                    {
                        "Method": r["Method"],
                        "Fusion Type": label,
                        "DG Strategy": "none",
                        "Uses Target": False,
                        "CIN2+ AUC (95% CI)": C.fmt(r["CIN2+ AUC"]),
                        "CIN3+ Sensitivity": C.fmt(r["CIN3+ sensitivity"]),
                        "CIN3+ FN": r["CIN3+ FN"],
                        "Centre Gap": C.fmt(r["Centre gap"]),
                        "Comparability Status": r.get("Comparability status", "LOCKED_LOCO_COMPARABLE"),
                        "Notes": "Existing locked LOCO prediction output; included with audit caveat.",
                    }
                )
    out = pd.DataFrame(rows).drop_duplicates(subset=["Method", "Fusion Type"], keep="first")
    C.write_table(out, "Table_Fusion_Ladder_IF_final")
    C.append_manifest("E02_FINAL", "Fusion Ladder Finalisation", "COMPLETED_WITH_CAVEAT", ["tables/Table_Fusion_Ladder_IF_final.csv"], "Rows retain comparability flags.")


def main() -> None:
    C.ensure_dirs()
    e30_clinical_baselines()
    e31_e32_e33_existing_baselines()
    final_ladder()


if __name__ == "__main__":
    main()
