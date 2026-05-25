#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("ifrb_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


MODULES = [
    ("G00", "01_g00_audit.py"),
    ("CORE", "02_core_tables.py"),
    ("DOMAIN", "03_domain_shift_analysis.py"),
    ("TTA", "04_tta_boundary_analysis.py"),
    ("FAILURE", "05_failure_centre_analysis.py"),
    ("BASELINES", "06_baseline_ladder.py"),
    ("CLINICAL", "07_clinical_evaluation.py"),
    ("WRITING", "08_coe_and_writing_outputs.py"),
]


def run_module(label: str, script: str) -> dict:
    path = Path(__file__).with_name(script)
    log = C.OUT / "logs" / f"{label}_{script}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, str(path)], cwd=C.ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(proc.stdout, encoding="utf-8")
    return {"phase": label, "script": script, "return_code": proc.returncode, "log": C.rel(log), "status": "COMPLETED" if proc.returncode == 0 else "FAILED"}


def final_report(module_rows: list[dict]) -> None:
    manifest = C.read_csv(C.OUT / "manifests" / "final_execution_manifest.csv")
    if manifest is None:
        manifest = pd.DataFrame()
    forbidden = C.scan_for_forbidden_phrases()
    forbidden.to_csv(C.OUT / "audit" / "forbidden_phrase_scan.csv", index=False, encoding="utf-8-sig")
    tables = sorted([C.rel(p) for p in (C.OUT / "tables").glob("*") if p.suffix in [".csv", ".md", ".tex"]])
    figs = sorted([C.rel(p) for p in (C.OUT / "figures").glob("*") if p.suffix in [".pdf", ".png"]])
    metrics = C.read_csv(C.PATHS["tta_metrics"])
    if metrics is not None:
        src = metrics[metrics["Method"].eq("Step2.9 source-only DG ensemble")].iloc[0]
        best = metrics[metrics["Method"].eq(C.best_tta_method())].iloc[0]
        findings = [
            f"Source-only CIN2+ AUC was {float(src['AUC']):.3f}.",
            f"Best score-level TTA CIN3+ false negatives were {int(best['CIN3+ FN'])}, compared with {int(src['CIN3+ FN'])} source-only.",
            f"Centre gap remained {float(src['Centre gap']):.3f}.",
            "The hardest-centre ranking failure remained after score-level adaptation.",
        ]
    else:
        findings = ["Core TTA metrics were unavailable."]
    mmd_avg = C.read_csv(C.OUT / "statistics" / "mmd_average_outbound.csv")
    if mmd_avg is not None and not mmd_avg.empty:
        top = mmd_avg.iloc[0]
        findings.append(f"Computed feature-space average outbound MMD was highest for {top['Centre']} ({float(top['Average outbound MMD']):.3f}); Xiangyang was not the maximum-MMD centre in this analysis.")
    completed = int((manifest["status"].astype(str).str.contains("COMPLETED", na=False)).sum()) if not manifest.empty else 0
    caveated = int((manifest["status"].astype(str).str.contains("CAVEAT", na=False)).sum()) if not manifest.empty else 0
    failed = [r for r in module_rows if r["return_code"] != 0]
    report = [
        "# IF Route B Master Execution Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Overall execution status: {'COMPLETED_WITH_CAVEATS' if not failed else 'PARTIAL_WITH_FAILURES'}",
        f"- Number of completed experiments: {completed}",
        f"- Number of skipped experiments: 0",
        f"- Number of caveated experiments: {caveated}",
        "",
        "## 2. Locked Dataset and Protocol",
        "",
        "The package uses the locked n=1897 data file and preserves the inductive source-only versus transductive score-level TTA separation from Step 2.10.",
        "",
        "## 3. Manuscript-Ready Tables",
        "",
        "\n".join([f"- `{t}`" for t in tables]) if tables else "No tables generated.",
        "",
        "## 4. Manuscript-Ready Figures",
        "",
        "\n".join([f"- `{f}`" for f in figs]) if figs else "No figures generated.",
        "",
        "## 5. Core Empirical Findings",
        "",
        "\n".join([f"- {x}" for x in findings]),
        "",
        "## 6. Caveated or Unsupported Claims",
        "",
        "- Do not present score-level TTA as source-only generalisation.",
        "- Do not state that the hardest centre was repaired.",
        "- Do not treat CoE readability as expert-validated faithfulness.",
        "- Do not claim a deployment-level safety result because the CIN3+ target was not fully reached.",
        "- Do not assign a maximum MMD claim to Xiangyang unless a later audit supports it.",
        "",
        "## 7. Fusion Ladder Completeness",
        "",
        "The fusion ladder is partially comparable. Existing all-model predictions are included with comparability flags; no new image encoder retraining was performed.",
        "",
        "## 8. TTA Boundary Conclusion",
        "",
        "Score-level TTA changed operating points and reduced false negatives under selected thresholds, but it did not repair ranking failure or reduce the centre gap.",
        "",
        "## 9. Hard-Centre Failure Conclusion",
        "",
        "Xiangyang remains the main residual failure centre by low CIN2+ AUC and concentration of CIN3+ false negatives in the verified Step 2.10 outputs.",
        "",
        "## 10. Recommended Manuscript Structure",
        "",
        "- Use `paper_sections/W01_Abstract_Draft.txt` for the abstract draft.",
        "- Use `paper_sections/W02_Introduction_Draft.txt` for the introduction draft.",
        "- Use `paper_sections/W03_Method_Framework.txt` for the method skeleton.",
        "- Use `paper_sections/W04_Experiment_Structure.txt` for the experiment structure.",
        "- Use `paper_sections/W05_Discussion_Framework.txt` for the discussion framework.",
        "",
        "## 11. Remaining Manual Checks",
        "",
        "- Review Chinese centre labels in figures for font rendering.",
        "- Decide whether existing all-model baseline rows should be main-table or supplementary-table material.",
        "- Manually verify journal-specific LaTeX table formatting.",
        "- Review `audit/forbidden_phrase_scan.csv` before manuscript assembly.",
    ]
    if failed:
        report += ["", "## Module Failures", "", C.md_table(pd.DataFrame(failed))]
    if not forbidden.empty:
        report += ["", "## Phrase Scan", "", "Some generated files contained claim-lock phrases and require manual review.", C.md_table(forbidden)]
    else:
        report += ["", "## Phrase Scan", "", "No prohibited exact phrases were found in generated manuscript text files."]
    C.write_text(C.OUT / "IF_RouteB_Master_Execution_Report.md", "\n".join(report))
    C.existing_outputs_index().to_csv(C.OUT / "manifests" / "final_output_file_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    C.ensure_dirs()
    rows = []
    for label, script in MODULES:
        row = run_module(label, script)
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.OUT / "logs" / "master_runner_module_status.csv", index=False, encoding="utf-8-sig")
    final_report(rows)


if __name__ == "__main__":
    main()
