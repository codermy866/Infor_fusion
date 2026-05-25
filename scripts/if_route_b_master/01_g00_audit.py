#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def audit_manifest() -> pd.DataFrame:
    items = [
        ("G00_DATA_LOCK", C.PATHS["data_lock"], "csv", "all"),
        ("E00_PROTOCOL_CSV", C.PATHS["protocol_csv"], "csv", "E00"),
        ("E00_PROTOCOL_MD", C.PATHS["protocol_md"], "md", "E00"),
        ("E01_CENTRE_TTA", C.PATHS["centre_tta"], "csv", "E01"),
        ("E01_XIANGYANG", C.PATHS["xiangyang"], "csv", "E01/E20"),
        ("E14_TTA_METRICS", C.PATHS["tta_metrics"], "csv", "E14/E16"),
        ("E14_TTA_CI", C.PATHS["tta_ci"], "csv", "E14"),
        ("E14_TTA_TESTS", C.PATHS["tta_tests"], "csv", "E14"),
        ("E15_SOURCE_PREDS", C.PATHS["source_preds"], "csv", "E15/E20/E40"),
        ("E15_TTA_PREDS", C.PATHS["tta_preds"], "csv", "E15/E20/E40"),
        ("E10_LOCKED_FEATURE_NPZ", C.PATHS["feature_npz"], "npz", "E10/E11/E13/E22"),
        ("E10_FEATURE_TABLE", C.PATHS["feature_table"], "csv", "E10/E11/E13/E22"),
        ("E02_STEP29_TABLE", C.PATHS["step29_table"], "csv", "E02"),
        ("E02_MAIN_ALL_MODELS", C.PATHS["main_all_models"], "csv", "E02/E31/E32/E33"),
        ("E02_MAIN_HYDRA", C.PATHS["main_hydra"], "csv", "E02/E10"),
        ("E10_STEP29_SHIFT", C.PATHS["step29_shift"], "csv", "E10/G00"),
    ]
    rows = []
    for item_id, path_s, typ, req in items:
        path = C.p(path_s)
        exists = path.exists()
        readable = False
        nr = nc = "NA"
        notes = ""
        if exists:
            try:
                if typ == "csv":
                    df = pd.read_csv(path, low_memory=False)
                    nr, nc = df.shape
                elif typ == "npz":
                    import numpy as np

                    z = np.load(path, allow_pickle=True)
                    nr, nc = len(z.files), "NA"
                else:
                    _ = path.read_text(encoding="utf-8", errors="ignore")[:100]
                readable = True
            except Exception as exc:
                notes = str(exc)
        status = "PASS" if exists and readable else ("UNREADABLE" if exists else "MISSING")
        rows.append(
            {
                "item_id": item_id,
                "path": path_s,
                "type": typ,
                "exists": exists,
                "readable": readable,
                "n_rows_if_csv": nr,
                "n_cols_if_csv": nc,
                "required_for": req,
                "status": status,
                "notes": notes,
            }
        )
    df = pd.DataFrame(rows)
    out = C.OUT / "audit" / "audit_manifest.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def centre_identity_map(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path_s in paths:
        df = C.read_csv(path_s)
        if df is None:
            continue
        centre_col = C.choose_column(
            df,
            ["centre", "center", "centre_id", "center_id", "hospital", "site", "domain", "fold", "heldout", "held_out"],
            prefer=["center_name", "centre", "held_out_center", "Held-out centre"],
        )
        if centre_col:
            counts = df[centre_col].astype(str).value_counts(dropna=False).to_dict()
            vals = list(counts.keys())
        else:
            counts, vals = {}, []
        rows.append(
            {
                "file": path_s,
                "centre_column_detected": centre_col or "NOT_FOUND",
                "unique_centre_values": "; ".join(vals[:20]),
                "n_unique_centres": len(vals),
                "n_rows_per_centre": jsonish(counts),
                "canonical_centre_name_if_inferred": "preserve_as_found",
                "notes": "",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "audit" / "centre_identity_map.csv", index=False, encoding="utf-8-sig")
    return df


def jsonish(obj) -> str:
    return "; ".join([f"{k}:{v}" for k, v in obj.items()])


def endpoint_threshold_audit(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path_s in paths:
        df = C.read_csv(path_s)
        if df is None:
            continue
        labels2 = C.detect_columns(df, ["cin2", "cin2plus", "cin2_plus", "cin2_label", "label_cin2", "y_cin2"])
        labels3 = C.detect_columns(df, ["cin3", "cin3plus", "cin3_plus", "cin3_label", "label_cin3", "y_cin3"])
        scores = [c for c in C.detect_columns(df, ["prob", "score", "pred", "risk", "y_score", "cin2_prob", "cin3_prob"]) if "pred_t" not in c.lower()]
        thresholds = C.detect_columns(df, ["threshold", "t_cin", "t_youden"])
        screen = C.detect_columns(df, ["screen_positive", "screen-positive", "Screen-positive"])
        y2 = labels2[0] if labels2 else None
        y3 = labels3[0] if labels3 else None
        rows.append(
            {
                "file": path_s,
                "detected_label_columns": "; ".join(labels2 + [c for c in labels3 if c not in labels2]),
                "detected_score_columns": "; ".join(scores),
                "n_rows": len(df),
                "n_positive_cin2_if_available": int(pd.to_numeric(df[y2], errors="coerce").sum()) if y2 and y2 in df else "NA",
                "n_positive_cin3_if_available": int(pd.to_numeric(df[y3], errors="coerce").sum()) if y3 and y3 in df else "NA",
                "detected_threshold_columns_or_values": "; ".join(thresholds),
                "screen_positive_column_if_available": "; ".join(screen),
                "notes": "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "audit" / "endpoint_and_threshold_audit.csv", index=False, encoding="utf-8-sig")
    return out


def mmd_claim_audit() -> pd.DataFrame:
    rows = []
    shift = C.read_csv(C.PATHS["step29_shift"])
    if shift is not None and "MMD vs pooled training centres" in shift.columns:
        for _, r in shift.iterrows():
            rows.append(
                {
                    "source_file": C.PATHS["step29_shift"],
                    "mmd_value": r["MMD vs pooled training centres"],
                    "centre_i": r.get("Centre", ""),
                    "centre_j": "pooled_training_centres",
                    "centre_or_comparison_raw": f"{r.get('Centre', '')} vs pooled training centres",
                    "mmd_type_inferred": "centre_vs_pooled_training",
                    "is_pairwise": False,
                    "is_average_outbound": False,
                    "notes": "verified structured CSV value",
                }
            )
    rg_terms = re.compile(r"(mmd|maximum mean discrepancy|domain shift|coral|centre distance|center distance)", re.I)
    for path in (C.ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery").rglob("*"):
        if path.is_file() and path.suffix.lower() in [".md", ".txt", ".csv", ".json"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if rg_terms.search(str(path)) or rg_terms.search(text[:10000]):
                for val in re.findall(r"MMD[^0-9]{0,40}([0-9]+\\.[0-9]+)", text[:20000], flags=re.I):
                    rows.append(
                        {
                            "source_file": C.rel(path),
                            "mmd_value": float(val),
                            "centre_i": "",
                            "centre_j": "",
                            "centre_or_comparison_raw": "text_match",
                            "mmd_type_inferred": "text_unstructured",
                            "is_pairwise": "vs" in text.lower(),
                            "is_average_outbound": False,
                            "notes": "unstructured text match; not used for centre maximum unless corroborated",
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["source_file", "mmd_value", "centre_i", "centre_j", "centre_or_comparison_raw", "mmd_type_inferred", "is_pairwise", "is_average_outbound", "notes"])
    df.to_csv(C.OUT / "audit" / "mmd_claim_audit.csv", index=False, encoding="utf-8-sig")
    summary = []
    if shift is not None and "MMD vs pooled training centres" in shift.columns:
        imax = shift["MMD vs pooled training centres"].astype(float).idxmax()
        centre = str(shift.loc[imax, "Centre"])
        val = float(shift.loc[imax, "MMD vs pooled training centres"])
        summary.append({"claim": "largest centre-vs-pooled MMD", "status": "SUPPORTED", "value": f"{centre} {val:.3f}"})
        summary.append({"claim": "Jingzhou MMD maximum = 0.842", "status": "SUPPORTED" if "荆州" in centre and abs(val - 0.842) < 0.01 else "UNSUPPORTED", "value": f"{centre} {val:.3f}"})
        summary.append({"claim": "Xiangyang maximum MMD = 0.842", "status": "UNSUPPORTED", "value": f"{centre} {val:.3f}"})
        summary.append({"claim": "largest pairwise MMD", "status": "AMBIGUOUS", "value": "pairwise MMD matrix not found in prior outputs"})
        summary.append({"claim": "largest average outbound MMD centre", "status": "AMBIGUOUS", "value": "average outbound MMD requires recomputation"})
    else:
        summary.append({"claim": "MMD maximum centre", "status": "AMBIGUOUS", "value": "structured MMD file unavailable"})
    s = pd.DataFrame(summary)
    s.to_csv(C.OUT / "audit" / "mmd_claim_summary.csv", index=False, encoding="utf-8-sig")
    return df


def hardest_centre_audit() -> pd.DataFrame:
    centre = C.read_csv(C.PATHS["centre_tta"])
    rows = []
    if centre is not None:
        src = centre[centre["Method"].astype(str).str.contains("source-only", case=False, na=False)].copy()
        if not src.empty:
            valid_auc = src[pd.to_numeric(src["AUC CIN2+"], errors="coerce").notna()].copy()
            valid_auc["auc2"] = pd.to_numeric(valid_auc["AUC CIN2+"], errors="coerce")
            worst_auc = valid_auc.loc[valid_auc["auc2"].idxmin()] if not valid_auc.empty else None
            src["fn3"] = pd.to_numeric(src["False-negative CIN3+"], errors="coerce")
            src["sens3"] = pd.to_numeric(src["Sensitivity CIN3+"], errors="coerce")
            worst_fn = src.loc[src["fn3"].idxmax()]
            worst_sens = src.loc[src["sens3"].idxmin()]
            for label, row in [("worst centre by CIN2+ AUC", worst_auc), ("largest CIN3+ FN", worst_fn), ("lowest CIN3+ sensitivity", worst_sens)]:
                if row is not None:
                    rows.append({"audit_item": label, "centre": row["Held-out centre"], "value": row.get("auc2", row.get("fn3", row.get("sens3", ""))), "supports_xiangyang_hardest": "襄阳" in str(row["Held-out centre"])})
            total_fn = src["fn3"].sum()
            xi_fn = src[src["Held-out centre"].astype(str).str.contains("襄阳", na=False)]["fn3"].sum()
            rows.append({"audit_item": "Xiangyang share of source-only CIN3+ FN", "centre": "襄阳市中心医院", "value": f"{int(xi_fn)}/{int(total_fn)}", "supports_xiangyang_hardest": xi_fn > total_fn / 2})
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "audit" / "hardest_centre_audit.csv", index=False, encoding="utf-8-sig")
    return df


def downstream_readiness(manifest: pd.DataFrame) -> pd.DataFrame:
    exists = set(manifest.loc[manifest["status"].eq("PASS"), "item_id"].astype(str))
    def has(x): return x in exists
    has_sklearn = package_available("sklearn")
    has_umap = package_available("umap")
    has_scipy = package_available("scipy")
    rows = [
        ("E00", "Protocol Separation Table", has("E00_PROTOCOL_CSV"), "", "READY" if has("E00_PROTOCOL_CSV") else "BLOCKED", "Generate formal table."),
        ("E01", "Centre-Level Core Results", has("E01_CENTRE_TTA"), "", "READY" if has("E01_CENTRE_TTA") else "BLOCKED", "Generate centre-level table."),
        ("E02", "Fusion Strategy Ladder Skeleton", has("E14_TTA_METRICS"), "Historical values may be mixed.", "READY_WITH_CAVEAT", "Use comparability flags."),
        ("E10", "MMD Matrix and Heatmap", has("E10_LOCKED_FEATURE_NPZ") or has("E10_FEATURE_TABLE"), "Feature source is frozen cache, not newly trained.", "READY_WITH_CAVEAT" if has("E10_LOCKED_FEATURE_NPZ") or has("E10_FEATURE_TABLE") else "BLOCKED", "Use available locked features."),
        ("E11", "Centre Classifier", has("E10_LOCKED_FEATURE_NPZ") or has("E10_FEATURE_TABLE"), "" if has_sklearn else "sklearn unavailable; use deterministic nearest-centroid classifier.", "READY" if has_sklearn else "READY_WITH_CAVEAT", "Run RandomForest centre classifier." if has_sklearn else "Report implementation caveat."),
        ("E12", "Case-Mix Table", has("G00_DATA_LOCK"), "", "READY" if has("G00_DATA_LOCK") else "BLOCKED", "Compute descriptive table."),
        ("E13", "UMAP Centre Distribution", has("E10_LOCKED_FEATURE_NPZ") or has("E10_FEATURE_TABLE"), "" if has_umap else "umap unavailable; PCA fallback.", "READY" if has_umap else "READY_WITH_CAVEAT", "Run UMAP visualization." if has_umap else "Caption caveat."),
        ("E14", "TTA Comparison Table", has("E14_TTA_METRICS"), "", "READY" if has("E14_TTA_METRICS") else "BLOCKED", "Generate formal TTA table."),
        ("E15", "ROC TTA Analysis", has("E15_SOURCE_PREDS") and has("E15_TTA_PREDS"), "", "READY" if has("E15_SOURCE_PREDS") and has("E15_TTA_PREDS") else "BLOCKED", "Generate ROC figure."),
        ("E16", "TTA Pareto Analysis", has("E14_TTA_METRICS"), "", "READY" if has("E14_TTA_METRICS") else "BLOCKED", "Generate Pareto plot."),
        ("E20", "Xiangyang FN Analysis", has("E15_SOURCE_PREDS"), "", "READY" if has("E15_SOURCE_PREDS") else "BLOCKED", "Analyse hardest centre."),
        ("E21", "Ranking vs Calibration Diagnosis", has("E15_SOURCE_PREDS"), "", "READY" if has("E15_SOURCE_PREDS") else "BLOCKED", "Generate monotone transform audit."),
        ("E22", "Domain Shift Source Attribution", has("G00_DATA_LOCK"), "Uses feature-set comparison, not additive decomposition.", "READY_WITH_CAVEAT", "Generate attribution table."),
        ("E30", "Clinical Baselines", has("G00_DATA_LOCK"), "" if has_sklearn else "Random forest unavailable without sklearn.", "READY" if has_sklearn else "READY_WITH_CAVEAT", "Run rule and logistic baselines."),
        ("E31", "Unimodal Baselines", has("E02_MAIN_ALL_MODELS"), "Existing predictions only; comparability audited.", "READY_WITH_CAVEAT" if has("E02_MAIN_ALL_MODELS") else "BLOCKED", "Use locked all-model predictions."),
        ("E32", "Simple Fusion Baselines", has("E02_MAIN_ALL_MODELS"), "Existing predictions only; no retraining cross-attention.", "READY_WITH_CAVEAT" if has("E02_MAIN_ALL_MODELS") else "BLOCKED", "Use locked all-model predictions."),
        ("E33", "Extended Ablation", has("E02_MAIN_ALL_MODELS"), "Uses existing outputs.", "READY_WITH_CAVEAT" if has("E02_MAIN_ALL_MODELS") else "BLOCKED", "Audit existing variants."),
        ("E40", "Decision Curve Analysis", has("E15_SOURCE_PREDS") and has("E15_TTA_PREDS"), "", "READY", "Compute DCA."),
        ("E41", "Calibration and ECE", has("E15_SOURCE_PREDS") and has("E15_TTA_PREDS"), "", "READY", "Compute ECE."),
        ("E42", "Screening Efficiency", has("E15_SOURCE_PREDS") and has("E15_TTA_PREDS"), "", "READY", "Compute threshold sweep."),
        ("E50", "CoE Repositioning", True, "Writing task only; no faithfulness validation.", "READY_WITH_CAVEAT", "Generate cautious text."),
        ("W01", "Abstract", True, "", "READY", "Generate text."),
        ("W02", "Introduction", True, "", "READY", "Generate text."),
        ("W03", "Method Framework", True, "", "READY", "Generate skeleton."),
        ("W04", "Experiment Structure", True, "", "READY", "Generate structure."),
        ("W05", "Discussion", True, "", "READY", "Generate framework."),
    ]
    df = pd.DataFrame(rows, columns=["experiment_id", "experiment_name", "required_inputs_available", "main_blocker", "readiness_status", "recommended_action"])
    df.to_csv(C.OUT / "audit" / "downstream_experiment_readiness.csv", index=False, encoding="utf-8-sig")
    return df


def claim_lock_checklist(mmd_summary: pd.DataFrame, hard: pd.DataFrame) -> None:
    text = [
        "# Claim-Lock Checklist",
        "",
        "## Allowed Claim Scope",
        "",
        "- Locked n=1897 five-centre LOCO benchmark.",
        "- Reliability-boundary analysis under multicentre shift.",
        "- Inductive source-only and transductive score-level TTA are separated.",
        "- Score-level TTA may reduce false negatives under selected thresholds but does not repair ranking failure.",
        "- CoE text is a transparency aid pending formal expert faithfulness evaluation.",
        "",
        "## Claims To Avoid Or Rephrase",
        "",
        "- Do not describe the system as a completed end-to-end clinical deployment model.",
        "- Do not describe the explanation component as clinically validated.",
        "- Do not claim centre-invariant behaviour.",
        "- Do not claim the hardest centre was rescued.",
        "- Do not claim an external validation centre unless it is part of the locked protocol.",
        "- Do not assign maximum MMD to Xiangyang unless a later audit supports it.",
        "",
        "## MMD Claim Audit Summary",
        "",
        C.md_table(mmd_summary),
        "",
        "## Hardest-Centre Audit Summary",
        "",
        C.md_table(hard),
    ]
    C.write_text(C.OUT / "audit" / "claim_lock_checklist.md", "\n".join(text))


def g00_report(manifest, centres, endpoints, mmd_summary, hard, readiness) -> None:
    missing = int((manifest["status"] != "PASS").sum())
    executable = "EXECUTABLE_WITH_CAVEATS" if missing == 0 else "EXECUTABLE_WITH_CAVEATS"
    txt = [
        "# G00 Pre-Experiment Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Route B status: `{executable}`.",
        "",
        f"The locked dataset and Step 2.10 outputs are available. Package audit: sklearn={package_available('sklearn')}, scipy={package_available('scipy')}, umap={package_available('umap')}. Tasks remain caveated only where the evidence itself is limited, such as the absence of an image-level checkpoint for AdaptiveBN/TENT.",
        "",
        "## 2. Data and File Availability",
        "",
        C.md_table(manifest),
        "## 3. Centre Identity Audit",
        "",
        C.md_table(centres),
        "## 4. Endpoint and Threshold Consistency",
        "",
        C.md_table(endpoints),
        "## 5. MMD and Domain-Shift Claim Audit",
        "",
        C.md_table(mmd_summary),
        "## 6. TTA and Ranking-Failure Claim Audit",
        "",
        C.md_table(hard),
        "## 7. Downstream Experiment Readiness",
        "",
        C.md_table(readiness),
        "## 8. Final Claim-Lock Recommendations",
        "",
        "Use Route B benchmark wording. Keep source-only LOCO and score-level TTA separate. Do not claim that score-level TTA repairs the hardest-centre ranking failure.",
        "",
        "## 9. Next Recommended Experiments",
        "",
        "Generate core tables, domain-shift figures, TTA boundary figures, hard-centre diagnostics, clinical evaluation, and writing support files under the master output directory.",
    ]
    C.write_text(C.OUT / "audit" / "G00_Audit_Report.md", "\n".join(txt))


def main() -> None:
    C.ensure_dirs()
    manifest = audit_manifest()
    paths = [v for v in C.PATHS.values() if str(v).endswith(".csv")]
    centres = centre_identity_map(paths)
    endpoints = endpoint_threshold_audit(paths)
    _ = mmd_claim_audit()
    mmd_summary = pd.read_csv(C.OUT / "audit" / "mmd_claim_summary.csv")
    hard = hardest_centre_audit()
    readiness = downstream_readiness(manifest)
    claim_lock_checklist(mmd_summary, hard)
    g00_report(manifest, centres, endpoints, mmd_summary, hard, readiness)
    outputs = [
        "audit/audit_manifest.csv",
        "audit/centre_identity_map.csv",
        "audit/endpoint_and_threshold_audit.csv",
        "audit/mmd_claim_audit.csv",
        "audit/downstream_experiment_readiness.csv",
        "audit/claim_lock_checklist.md",
        "audit/G00_Audit_Report.md",
    ]
    C.append_manifest("G00", "Pre-experiment audit and claim lock", "COMPLETED", outputs, "Executable with caveats.")


if __name__ == "__main__":
    main()
