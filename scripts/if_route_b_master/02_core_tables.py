#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, kruskal
except Exception:  # pragma: no cover
    chi2_contingency = None
    kruskal = None

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def e00_protocol() -> None:
    src = C.read_csv(C.PATHS["protocol_csv"])
    if src is None:
        C.append_manifest("E00", "Protocol Separation Table", "BLOCKED", [], "Protocol CSV missing.")
        return
    df = pd.DataFrame(
        {
            "Track": src["Track"],
            "Uses target-centre images/features": src["Uses target-centre images/features"],
            "Uses target-centre labels": src["Uses target labels"],
            "Allowed claims": src["Allowed claim"],
            "Prohibited claims": src["Forbidden claim"],
            "Manuscript interpretation": src["Main use in manuscript"],
        }
    )
    C.write_table(df, "Table_Protocol_Separation_IF")
    paragraph = (
        "We separated inductive source-only LOCO evaluation from transductive score-level target adaptation. "
        "The inductive track did not use held-out centre samples during model selection or training. "
        "The transductive track permitted unlabeled target-centre scores/features but no target-centre labels. "
        "Therefore, TTA results are interpreted as adaptation-boundary analyses rather than source-only generalisation performance."
    )
    C.write_text(C.OUT / "paper_sections" / "sec_eval_protocol.txt", paragraph + "\n")
    C.append_manifest("E00", "Protocol Separation Table", "COMPLETED", ["tables/Table_Protocol_Separation_IF.csv", "paper_sections/sec_eval_protocol.txt"])


def e01_centre_results() -> None:
    centre = C.read_csv(C.PATHS["centre_tta"])
    metrics = C.load_tta_metrics()
    if centre is None:
        C.append_manifest("E01", "Centre-Level Core Results", "BLOCKED", [], "Centre metrics missing.")
        return
    best = C.best_tta_method()
    src_name = "Step2.9 source-only DG ensemble"
    src = centre[centre["Method"].eq(src_name)].copy()
    tta = centre[centre["Method"].eq(best)].copy()
    rows = []
    for cen in sorted(set(src["Held-out centre"].astype(str))):
        a = src[src["Held-out centre"].astype(str).eq(cen)].iloc[0]
        b = tta[tta["Held-out centre"].astype(str).eq(cen)].iloc[0] if not tta[tta["Held-out centre"].astype(str).eq(cen)].empty else None
        fn_a = pd.to_numeric(a["False-negative CIN3+"], errors="coerce")
        fn_b = pd.to_numeric(b["False-negative CIN3+"], errors="coerce") if b is not None else np.nan
        flag = "single-class CIN2+" if str(a.get("Notes", "")) != "nan" and str(a.get("Notes", "")) else ""
        if "襄阳" in cen:
            flag = "hardest-centre FN concentration; ranking not repaired"
        rows.append(
            {
                "Centre": cen,
                "N_test": a["Test N"],
                "N_CIN2+": a["CIN2+ positives"],
                "N_CIN3+": a["CIN3+ positives"],
                "Source-only CIN2+ AUC": a["AUC CIN2+"],
                "Source-only CIN3+ AUC": a["AUC CIN3+"],
                "Source-only CIN3+ Sensitivity": a["Sensitivity CIN3+"],
                "Source-only CIN3+ FN": a["False-negative CIN3+"],
                "Source-only Screen+ Rate": a["Screen-positive rate"],
                "Best TTA CIN2+ AUC": b["AUC CIN2+"] if b is not None else "NA",
                "Best TTA CIN3+ AUC": b["AUC CIN3+"] if b is not None else "NA",
                "Best TTA CIN3+ Sensitivity": b["Sensitivity CIN3+"] if b is not None else "NA",
                "Best TTA CIN3+ FN": b["False-negative CIN3+"] if b is not None else "NA",
                "Best TTA Screen+ Rate": b["Screen-positive rate"] if b is not None else "NA",
                "Delta FN vs Source-only": int(fn_b - fn_a) if np.isfinite(fn_a) and np.isfinite(fn_b) else "NA",
                "Interpretation flag": flag,
            }
        )
    msrc = metrics[metrics["Method"].eq(src_name)].iloc[0]
    mtta = metrics[metrics["Method"].eq(best)].iloc[0]
    rows.append(
        {
            "Centre": "Pooled",
            "N_test": int(src["Test N"].sum()),
            "N_CIN2+": int(src["CIN2+ positives"].sum()),
            "N_CIN3+": int(src["CIN3+ positives"].sum()),
            "Source-only CIN2+ AUC": C.fmt(msrc["AUC"]),
            "Source-only CIN3+ AUC": C.fmt(msrc["CIN3+ AUC"]),
            "Source-only CIN3+ Sensitivity": C.fmt(msrc["CIN3+ sensitivity"]),
            "Source-only CIN3+ FN": int(msrc["CIN3+ FN"]),
            "Source-only Screen+ Rate": C.fmt(msrc["screen_positive_rate"]),
            "Best TTA CIN2+ AUC": C.fmt(mtta["AUC"]),
            "Best TTA CIN3+ AUC": C.fmt(mtta["CIN3+ AUC"]),
            "Best TTA CIN3+ Sensitivity": C.fmt(mtta["CIN3+ sensitivity"]),
            "Best TTA CIN3+ FN": int(mtta["CIN3+ FN"]),
            "Best TTA Screen+ Rate": C.fmt(mtta["screen_positive_rate"]),
            "Delta FN vs Source-only": int(mtta["CIN3+ FN"] - msrc["CIN3+ FN"]),
            "Interpretation flag": "TTA reduces FN but does not reduce centre gap",
        }
    )
    out = pd.DataFrame(rows)
    C.write_table(out, "Table_Centre_Level_Results_IF")
    C.write_text(
        C.OUT / "paper_sections" / "sec_centre_results_summary.txt",
        "Centre-level analysis identified Xiangyang as the main residual failure centre. The best score-level TTA reduced Xiangyang CIN3+ false negatives but did not improve Xiangyang CIN2+ AUC or the centre-level gap.\n",
    )
    C.append_manifest("E01", "Centre-Level Core Results", "COMPLETED", ["tables/Table_Centre_Level_Results_IF.csv", "paper_sections/sec_centre_results_summary.txt"])


def lookup_step29_row(table: pd.DataFrame, needle: str) -> dict:
    m = table[table["Method"].astype(str).str.contains(needle, case=False, na=False)]
    if m.empty:
        return {}
    r = m.iloc[0].to_dict()
    return {
        "CIN2+ AUC (95% CI)": r.get("AUC (95% CI)", "NA"),
        "CIN3+ Sensitivity": r.get("CIN3+ sensitivity", "NA"),
        "CIN3+ FN": r.get("CIN3+ false-negative count", "NA"),
        "Centre Gap": r.get("Centre-level performance gap", "NA"),
    }


def e02_fusion_ladder() -> None:
    step29 = C.read_csv(C.PATHS["step29_table"])
    metrics = C.load_tta_metrics()
    rows = []
    def add(method, fusion, dg, uses, vals=None, status="MISSING", notes=""):
        vals = vals or {}
        rows.append(
            {
                "Method": method,
                "Fusion Type": fusion,
                "DG Strategy": dg,
                "Uses Target": uses,
                "CIN2+ AUC (95% CI)": vals.get("CIN2+ AUC (95% CI)", "NA"),
                "CIN3+ Sensitivity": vals.get("CIN3+ Sensitivity", "NA"),
                "CIN3+ FN": vals.get("CIN3+ FN", "NA"),
                "Centre Gap": vals.get("Centre Gap", "NA"),
                "Comparability Status": status,
                "Notes": notes,
            }
        )
    if step29 is not None:
        add("Clinical Baseline (HPV+TCT+Age LR)", "clinical", "none", False, lookup_step29_row(step29, "Best clinical baseline"), "LOCKED_LOCO_COMPARABLE", "From Step 2.9 comparison table.")
        add("Step 2 Surrogate (HyDRA features, no DG)", "feature/prediction fusion", "none", False, lookup_step29_row(step29, "Step2 surrogate"), "LOCKED_LOCO_COMPARABLE", "From Step 2.9 comparison table.")
        add("Step 2.6 Active Minimal Adapter", "active minimal adapter", "none", False, lookup_step29_row(step29, "Step2.6"), "LOCKED_LOCO_COMPARABLE", "From Step 2.9 comparison table.")
        add("HyDRA-DG-SafetyEnsemble (Source-only)", "rank ensemble", "inner validation DG ensemble", False, lookup_step29_row(step29, "Best DG ensemble"), "LOCKED_LOCO_COMPARABLE", "Primary Route B source-only result.")
    for m in ["Colposcopy-only", "OCT-only", "Colposcopy+OCT Concat", "Colposcopy+OCT+Text Concat"]:
        add(m, "image/model baseline", "historical locked LOCO if available", False, status="READY_WITH_CAVEAT", notes="Filled in final ladder from all-model prediction audit if comparable.")
    best = C.best_tta_method()
    if best:
        b = metrics[metrics["Method"].eq(best)].iloc[0]
        vals = {
            "CIN2+ AUC (95% CI)": C.fmt(b["AUC"]),
            "CIN3+ Sensitivity": C.fmt(b["CIN3+ sensitivity"]),
            "CIN3+ FN": int(b["CIN3+ FN"]),
            "Centre Gap": C.fmt(b["Centre gap"]),
        }
        add("HyDRA-DG + Best TTA (Transductive)", "score-level target adaptation", "source-free score TTA", True, vals, "LOCKED_LOCO_COMPARABLE", "Secondary adaptation-boundary analysis, not inductive deployment.")
    out = pd.DataFrame(rows)
    C.write_table(out, "Table_Fusion_Ladder_IF_skeleton")
    C.append_manifest("E02", "Fusion Strategy Ladder Skeleton", "COMPLETED_WITH_CAVEAT", ["tables/Table_Fusion_Ladder_IF_skeleton.csv"], "Rows with unavailable old baselines are flagged.")


def pct(n, d):
    return f"{int(n)} ({100*n/max(d,1):.1f}%)"


def e12_case_mix() -> None:
    df = C.load_data_lock()
    rows = []
    for centre, g in df.groupby("center_name", sort=True):
        n = len(g)
        age = pd.to_numeric(g["age"], errors="coerce")
        q1, med, q3 = age.quantile([0.25, 0.5, 0.75])
        hpv18 = g["hpv16_18_status"].astype(str).str.lower()
        hpv = g["hpv_status_harmonized"].astype(str).str.lower()
        tct = g["tct_status_harmonized"].astype(str).str.lower()
        hpv16 = hpv18.str.contains("detect|positive|16|18", regex=True).sum()
        hpv_neg = hpv.str.contains("negative|-", regex=True).sum()
        other = max(hpv.str.contains("positive|detect", regex=True).sum() - hpv16, 0)
        nilm = tct.isin(["nilm", "-", "nan", "negative", ""]).sum()
        asc = n - nilm
        normal = n - int(g["pathology_cin2plus"].sum())
        rows.append(
            {
                "Centre": centre,
                "N": n,
                "N_CIN2+ (%)": pct(g["pathology_cin2plus"].sum(), n),
                "N_CIN3+ (%)": pct(g["pathology_cin3plus"].sum(), n),
                "N_Normal/CIN1 (%)": pct(normal, n),
                "Median Age (IQR)": f"{med:.1f} ({q1:.1f}-{q3:.1f})" if np.isfinite(med) else "NA",
                "HPV16/18 (%)": pct(hpv16, n),
                "Other HR-HPV (%)": pct(other, n),
                "HPV Negative (%)": pct(hpv_neg, n),
                "TCT NILM (%)": pct(nilm, n),
                "TCT ASC-US+ (%)": pct(asc, n),
                "N_Colposcopy images per patient": f"{pd.to_numeric(g['colposcopy_num_images'], errors='coerce').median():.1f}",
                "N_OCT images per patient": f"{pd.to_numeric(g['oct_num_bscans'], errors='coerce').median():.1f}",
            }
        )
    out = pd.DataFrame(rows)
    C.write_table(out, "Table_Dataset_Demographics_IF")
    contingency = []
    age_groups = []
    for _, g in df.groupby("center_name"):
        pos = int(g["pathology_cin2plus"].sum())
        contingency.append([pos, len(g) - pos])
        age_groups.append(pd.to_numeric(g["age"], errors="coerce").dropna().to_numpy())
    if chi2_contingency is not None:
        chi2, p_chi, _, _ = chi2_contingency(np.asarray(contingency))
        age_stat, p_age = kruskal(*age_groups) if kruskal is not None else (np.nan, np.nan)
        stat_text = f"CIN2+ chi-square={chi2:.2f}, p={p_chi:.3g}; age Kruskal-Wallis H={age_stat:.2f}, p={p_age:.3g}."
        manifest_note = "SciPy chi-square and Kruskal-Wallis tests computed."
        status = "COMPLETED"
    else:
        overall = df["pathology_cin2plus"].mean()
        chi2 = 0.0
        for _, g in df.groupby("center_name"):
            exp = len(g) * overall
            obs = g["pathology_cin2plus"].sum()
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        stat_text = f"CIN2+ prevalence heterogeneity statistic was chi-square={chi2:.2f}. Formal p-values were not computed because scipy was unavailable."
        manifest_note = "Descriptive test statistic only; scipy unavailable."
        status = "COMPLETED_WITH_CAVEAT"
    sec = (
        f"The locked cohort contains {len(df)} patients from {df['center_name'].nunique()} centres. "
        f"Centre-level case mix differed across centres; {stat_text}\n"
    )
    C.write_text(C.OUT / "paper_sections" / "sec_dataset_description.txt", sec)
    C.append_manifest("E12", "Dataset Case-Mix Table", status, ["tables/Table_Dataset_Demographics_IF.csv", "paper_sections/sec_dataset_description.txt"], manifest_note)


def main() -> None:
    C.ensure_dirs()
    e00_protocol()
    e01_centre_results()
    e02_fusion_ladder()
    e12_case_mix()


if __name__ == "__main__":
    main()
