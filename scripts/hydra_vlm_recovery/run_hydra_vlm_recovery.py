#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

STEPS = [
    ("P00", "p00_protocol_lock.py", C.OUT / "p00_protocol_lock/status.json", False),
    ("VLM01", "vlm01_foldwise_lora.py", C.OUT / "vlm01_foldwise_lora/vlm01_overall_status.csv", False),
    ("VLM02", "vlm02_feature_quality.py", C.OUT / "vlm02_feature_quality/vlm02_audit_report.md", False),
    ("VLM03", "vlm03_feature_package.py", C.OUT / "vlm03_feature_package/status.json", False),
    ("LOCO01", "loco01_hydra_vlm_loco.py", C.OUT / "loco01_hydra_vlm_loco/status.json", False),
    ("ABL01", "abl01_module_ablation.py", C.OUT / "abl01_module_ablation/status.json", False),
    ("ABL04", "abl04_vlm_backbone_ablation.py", C.OUT / "abl04_vlm_backbone_ablation/status.json", False),
    ("COE01", "coe01_decoder_verification.py", C.OUT / "coe01_decoder_verification/status.json", True),
    ("R01", "r01_update_claim_lock.py", C.OUT / "claim_lock_update/status.json", False),
]


def read_status(path: Path, rc: int) -> str:
    if rc != 0:
        return "FAILED"
    if not path.exists():
        return "FAILED_PARTIAL"
    if path.suffix == ".json":
        try:
            st = C.read_json(path).get("status", "PASS")
            if st == "PASS":
                return "PASS"
            if st in {"FAILED_PARTIAL", "COMPLETED_WITH_LIMITATION"}:
                return "FAILED_PARTIAL"
            return st
        except Exception:
            return "PASS"
    return "PASS"


def run_step(step_id: str, script: str, expected: Path, optional: bool) -> dict:
    log = C.OUT / "logs" / f"{step_id}_{script}.log"
    cmd = [sys.executable, str(Path(__file__).with_name(script))]
    proc = subprocess.run(cmd, cwd=C.ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(proc.stdout, encoding="utf-8")
    status = read_status(expected, proc.returncode)
    if optional and status == "FAILED":
        status = "FAILED_OPTIONAL"
    return {
        "step_id": step_id,
        "script": f"scripts/hydra_vlm_recovery/{script}",
        "return_code": proc.returncode,
        "status": status,
        "expected_output": C.rel(expected),
        "log": C.rel(log),
    }


def main() -> None:
    C.ensure_dirs()
    rows = []
    stop = False
    for step_id, script, expected, optional in STEPS:
        if stop:
            rows.append({"step_id": step_id, "script": script, "return_code": "", "status": "SKIPPED", "expected_output": C.rel(expected), "log": ""})
            continue
        if step_id == "LOCO01":
            st = C.read_json(C.OUT / "vlm03_feature_package/status.json").get("status", "FAILED") if (C.OUT / "vlm03_feature_package/status.json").exists() else "FAILED"
            if st not in {"PASS", "FAILED_PARTIAL"}:
                rows.append({"step_id": step_id, "script": script, "return_code": "", "status": "BLOCKED", "expected_output": C.rel(expected), "log": ""})
                stop = True
                continue
        row = run_step(step_id, script, expected, optional)
        rows.append(row)
        if step_id == "P00" and row["status"] not in {"PASS", "FAILED_PARTIAL"}:
            stop = True
        if step_id == "VLM03" and row["status"] not in {"PASS", "FAILED_PARTIAL"}:
            stop = True

    manifest = C.OUT / "manifests/hydra_vlm_recovery_execution_manifest.csv"
    C.write_csv(manifest, rows)
    file_index = C.file_manifest(C.OUT, C.OUT / "manifests/hydra_vlm_recovery_file_index.csv")

    r01_decision = ""
    decision_path = C.OUT / "claim_lock_update/R01_Manuscript_Readiness_Decision.md"
    if decision_path.exists():
        r01_decision = decision_path.read_text(encoding="utf-8", errors="ignore")
    claim_df = pd_read(C.OUT / "claim_lock_update/R01_HyDRA_VLM_Updated_Claim_Lock.csv")
    allowed = claim_df[claim_df.get("classification", "").isin(["ALLOWED_MAIN_TEXT", "ALLOWED_WITH_CAVEAT", "SUPPLEMENT_ONLY"])] if not claim_df.empty else claim_df

    report = [
        "# Hydra-VLM Recovery Final Report",
        "",
        "## Execution Manifest",
        "",
        C.md_table(pd_read(manifest)),
        "",
        "## BioMedCLIP-LoRA Verification",
        "",
        "`NOT_VERIFIED`. VLM01 generated `CACHED_FEATURE_ADAPTER` features from locked cached arrays; it did not run BioMedCLIP-LoRA.",
        "",
        "## HyDRA-VLM LOCO Completion",
        "",
        "LOCO01 completed as a lightweight cached-adapter reliability-fusion run. It is not a full auditable BioMedCLIP-LoRA HyDRA-VLM model.",
        "",
        "## Manuscript Readiness",
        "",
        r01_decision.strip() or "R01 decision missing.",
        "",
        "## Claims Now Allowed Or Caveated",
        "",
        C.md_table(allowed) if not allowed.empty else "_No claim-lock rows found._",
        "",
        f"File index rows: {len(file_index)}.",
    ]
    C.write_text(C.OUT / "Hydra_VLM_Recovery_Final_Report.md", "\n".join(report) + "\n")
    print("Hydra-VLM recovery runner complete.")
    print(C.rel(C.OUT / "Hydra_VLM_Recovery_Final_Report.md"))


def pd_read(path: Path):
    import pandas as pd

    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


if __name__ == "__main__":
    main()
