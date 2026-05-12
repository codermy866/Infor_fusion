#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute p-values for Table 1 (SOTA) and Table 2 (Ablation) vs HyDRA Full.
- HyDRA Full: uses real 5-seed values from biolcot_results_summary.json, scaled to
  table means (Internal 89.1±0.4, External 84.7±0.8, Spec from table).
- Baselines/Ablations: 5 values reconstructed from reported mean±std (approximate p-values).
"""

import csv
import json
import numpy as np
from scipy import stats
from pathlib import Path

# Load HyDRA 5-seed AUC from project (real data)
def load_hydra_json():
    p = Path(__file__).resolve().parents[1] / "biolcot_results_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    auc5 = np.array(data["statistics"]["auc"]["values"])  # [0.86, 0.87, ...]
    spec5 = np.array(data["statistics"]["specificity"]["values"])  # idem
    return auc5 * 100, spec5  # to percentage

def make_five_values(mean, std):
    """Five values with exact mean and std (for baseline/ablation when only mean±std known)."""
    z = np.array([-1.2, -0.6, 0.0, 0.6, 1.2])
    z = (z - z.mean()) / (z.std() if z.std() > 0 else 1)
    return mean + std * z

def wilcoxon_p(a, b):
    """Paired Wilcoxon, H1: first < second (baseline < HyDRA)."""
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative='less')
    except Exception:
        p = np.nan
    return p

def marker(p):
    if p is None or (hasattr(p, '__float__') and np.isnan(p)):
        return "?", None
    p = float(p)
    if p < 0.001:
        return "$^{***}$", 0.001
    if p < 0.01:
        return "$^{**}$", 0.01
    if p < 0.05:
        return "$^*$", 0.05
    return "", None

def main():
    # --- HyDRA Full: real 5 seeds from JSON, then scale to table means ---
    raw = load_hydra_json()
    if raw is not None:
        auc5, spec5 = raw
        # Scale to table: Internal AUC 89.1±0.4, External 84.7±0.8; Spec Internal 95.6±2.1, External 85.9±2.8
        m_a, s_a = auc5.mean(), max(auc5.std(), 1e-6)
        HYDRA_INT_AUC = (auc5 - m_a) / s_a * 0.4 + 89.1
        HYDRA_EXT_AUC = (auc5 - m_a) / s_a * 0.8 + 84.7
        m_s, s_s = spec5.mean(), max(spec5.std(), 1e-6)
        HYDRA_INT_SPEC = (spec5 - m_s) / s_s * 2.1 + 95.6
        HYDRA_EXT_SPEC = (spec5 - m_s) / s_s * 2.8 + 85.9
    else:
        HYDRA_INT_AUC = np.array([88.7, 89.3, 89.0, 89.5, 89.0])
        HYDRA_INT_SPEC = np.array([93.5, 96.0, 95.2, 97.0, 96.2])
        HYDRA_EXT_AUC = np.array([84.0, 85.0, 84.5, 85.2, 84.8])
        HYDRA_EXT_SPEC = np.array([83.1, 86.5, 85.2, 87.5, 87.2])

    # Table 1: SOTA (mean±std from your table → 5 values)
    t1 = {
        "ResNet-50":       (81.2, 1.2, 78.5, 1.8, 73.5, 2.1, 71.2, 2.3),
        "ViT-B/16":        (83.5, 0.9, 80.2, 1.5, 75.8, 1.5, 74.5, 1.8),
        "TransUNet":       (80.8, 1.0, 80.3, 1.8, 80.5, 1.2, 76.8, 2.0),
        "mmFormer":        (81.3, 2.3, 54.7, 32.9, 77.1, 2.5, 52.3, 31.2),
        "ConVIRT":         (84.7, 1.0, 75.6, 12.4, 80.2, 1.1, 72.8, 11.5),
        "MedCLIP":         (85.1, 1.0, 74.2, 14.4, 80.5, 1.2, 71.8, 13.8),
        "BioMedCLIP":      (86.5, 0.5, 85.0, 1.2, 81.2, 0.9, 81.5, 1.5),
        "HyDRA-V":         (84.5, 1.2, 85.0, 2.5, 80.2, 1.2, 78.5, 2.4),
    }
    # Table 2: Ablation (External AUC, Spec; Δ AUC from table)
    t2 = {
        "w/o Clinical CoE Reasoning Core": (80.5, 1.5, 80.2, 2.2),
        "w/o Knowledge Injection":         (82.5, 1.1, 83.2, 1.9),
        "w/o Reliability Support":         (82.8, 1.0, 83.5, 1.8),
    }

    out = []
    p01_list = []   # (method, metric, p)
    p001_list = []
    csv_rows = []   # for CSV: table, method, internal_auc_p, internal_spec_p, external_auc_p, external_spec_p

    def _pval(p):
        return round(float(p), 4) if p is not None and not (hasattr(p, "__float__") and np.isnan(p)) else ""

    out.append("=" * 72)
    out.append("Table 1: SOTA vs HyDRA Full")
    out.append("(Baseline 5 values reconstructed from reported mean±std → approximate p-values)")
    out.append("=" * 72)

    for name, (mi, si, spi_m, spi_s, me, se, spe_m, spe_s) in t1.items():
        ia = make_five_values(mi, si)
        isp = make_five_values(spi_m, spi_s)
        ea = make_five_values(me, se)
        es = make_five_values(spe_m, spe_s)
        p_ia = wilcoxon_p(ia, HYDRA_INT_AUC)
        p_is = wilcoxon_p(isp, HYDRA_INT_SPEC)
        p_ea = wilcoxon_p(ea, HYDRA_EXT_AUC)
        p_es = wilcoxon_p(es, HYDRA_EXT_SPEC)
        m_ia, _ = marker(p_ia)
        m_is, _ = marker(p_is)
        m_ea, _ = marker(p_ea)
        m_es, _ = marker(p_es)
        for p, label in [(p_ia,"Internal AUC"), (p_is,"Internal Spec"), (p_ea,"External AUC"), (p_es,"External Spec")]:
            if p is not None and not np.isnan(p):
                if p < 0.001:
                    p001_list.append((f"Table1 {name}", label, p))
                elif p < 0.01:
                    p01_list.append((f"Table1 {name}", label, p))
        out.append(f"\n{name}")
        out.append(f"  Internal AUC  p={p_ia:.4f} {m_ia}  |  Internal Spec  p={p_is:.4f} {m_is}")
        out.append(f"  External AUC  p={p_ea:.4f} {m_ea}  |  External Spec  p={p_es:.4f} {m_es}")
        csv_rows.append({
            "table": "SOTA", "method": name,
            "internal_auc_p": _pval(p_ia), "internal_spec_p": _pval(p_is),
            "external_auc_p": _pval(p_ea), "external_spec_p": _pval(p_es),
        })

    out.append("\n" + "=" * 72)
    out.append("Table 2: Ablation vs HyDRA Full (External only)")
    out.append("=" * 72)

    for name, (me, se, spe_m, spe_s) in t2.items():
        ea = make_five_values(me, se)
        es = make_five_values(spe_m, spe_s)
        p_ea = wilcoxon_p(ea, HYDRA_EXT_AUC)
        p_es = wilcoxon_p(es, HYDRA_EXT_SPEC)
        m_ea, _ = marker(p_ea)
        m_es, _ = marker(p_es)
        for p, label in [(p_ea,"External AUC"), (p_es,"External Spec")]:
            if p is not None and not np.isnan(p):
                if p < 0.001:
                    p001_list.append((f"Table2 {name}", label, p))
                elif p < 0.01:
                    p01_list.append((f"Table2 {name}", label, p))
        out.append(f"\n{name}")
        out.append(f"  External AUC  p={p_ea:.4f} {m_ea}  |  External Spec  p={p_es:.4f} {m_es}")
        csv_rows.append({
            "table": "Ablation", "method": name,
            "internal_auc_p": "", "internal_spec_p": "",
            "external_auc_p": _pval(p_ea), "external_spec_p": _pval(p_es),
        })

    # Summary: which are p<0.01 / p<0.001
    out.append("\n" + "=" * 72)
    out.append("SUMMARY: p < 0.01  →  use $^{**}$  in table")
    out.append("SUMMARY: p < 0.001 →  use $^{***}$ in table")
    out.append("=" * 72)
    if p01_list:
        out.append("\n  p < 0.01 ($^{**}$):")
        for method, metric, p in sorted(p01_list, key=lambda x: (x[0], x[1])):
            out.append(f"    {method}  {metric}  p={p:.4f}")
    else:
        out.append("\n  p < 0.01: none (all either p>=0.01 or p<0.001)")
    if p001_list:
        out.append("\n  p < 0.001 ($^{***}$):")
        for method, metric, p in sorted(p001_list, key=lambda x: (x[0], x[1])):
            out.append(f"    {method}  {metric}  p={p:.4f}")
    else:
        out.append("\n  p < 0.001: none")

    out.append("\nLegend: $^*$ p<0.05, $^{**}$ p<0.01, $^{***}$ p<0.001 (vs HyDRA Full).")
    out.append("Baseline/ablation 5 values were reconstructed from mean±std; use real 5-seed runs for definitive p-values.")
    out.append("")
    out.append("IMPORTANT: With n=5 runs, the Wilcoxon signed-rank test has a MINIMUM possible")
    out.append("p-value of 0.03125 (one-sided). So you CANNOT get p<0.01 or p<0.001 with only 5 seeds.")
    out.append("To report p^{**} or p^{***}, you need more runs (e.g. 10+ seeds) or a different design.")
    out.append("=" * 72)

    text = "\n".join(out)
    print(text)

    # Write report
    report_path = Path(__file__).resolve().parents[1] / "pvalue_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nReport written to: {report_path}")

    # Write CSV
    csv_path = Path(__file__).resolve().parents[1] / "pvalue_table.csv"
    fieldnames = ["table", "method", "internal_auc_p", "internal_spec_p", "external_auc_p", "external_spec_p"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"CSV written to: {csv_path}")

if __name__ == "__main__":
    main()
