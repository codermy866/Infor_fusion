#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "claim_lock_update"


def exists(rel: str) -> bool:
    return (C.OUT / rel).exists()


def status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        return C.read_json(path).get("status", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def main() -> None:
    C.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    vlm01_status = pd.read_csv(C.OUT / "vlm01_foldwise_lora/vlm01_overall_status.csv") if exists("vlm01_foldwise_lora/vlm01_overall_status.csv") else pd.DataFrame()
    lora_verified = (not vlm01_status.empty) and bool(vlm01_status["biomedclip_lora_verified"].fillna(False).all())
    loco_status = status(C.OUT / "loco01_hydra_vlm_loco/status.json")
    abl01_status = status(C.OUT / "abl01_module_ablation/status.json")
    abl04_status = status(C.OUT / "abl04_vlm_backbone_ablation/status.json")
    coe_status = status(C.OUT / "coe01_decoder_verification/status.json")
    routeb_tta = C.ROUTE_B_TTA.exists()
    routeb_centre = C.ROUTE_B_CENTRE.exists()

    loco_decision = "NOT_AVAILABLE"
    if exists("loco01_hydra_vlm_loco/model_config.json"):
        loco_decision = C.read_json(C.OUT / "loco01_hydra_vlm_loco/model_config.json").get("HYDRA_VLM_IMPROVES_OVER_ROUTE_B", "UNKNOWN")
    abl04_decision = "NOT_AVAILABLE"
    if exists("abl04_vlm_backbone_ablation/status.json"):
        abl04_decision = C.read_json(C.OUT / "abl04_vlm_backbone_ablation/status.json").get("BIO_MEDCLIP_LORA_SUPPORTED", "NOT_TESTABLE")

    rows = [
        {"claim_id": "C01", "claim": "Locked n=1897 five-centre LOCO benchmark.", "classification": "ALLOWED_MAIN_TEXT", "support": "P00 protocol lock and Route B outputs.", "safe_wording": "locked n=1897 five-centre LOCO benchmark", "notes": ""},
        {"claim_id": "C02", "claim": "Fold-wise VLM-LoRA was completed without target-label leakage.", "classification": "NOT_VERIFIED" if not lora_verified else "ALLOWED_MAIN_TEXT", "support": "VLM01", "safe_wording": "[TO VERIFY] or state cached-feature adapter only", "notes": "BioMedCLIP-LoRA not run in recovery."},
        {"claim_id": "C03", "claim": "VLM-LoRA improves feature quality over frozen BioMedCLIP.", "classification": "NOT_VERIFIED", "support": "VLM02/ABL04", "safe_wording": "cached feature analysis only", "notes": f"ABL04 decision: {abl04_decision}."},
        {"claim_id": "C04", "claim": "VLM-LoRA reduces centre shift/MMD.", "classification": "NOT_VERIFIED", "support": "VLM02/ABL04 MMD", "safe_wording": "MMD was evaluated for cached/frozen features", "notes": "No VLM-LoRA delta exists."},
        {"claim_id": "C05", "claim": "HyDRA-VLM improves pooled CIN2+ AUC over Route B HyDRA-DG.", "classification": "NOT_VERIFIED", "support": "LOCO01", "safe_wording": "cached-adapter LOCO run is exploratory", "notes": f"LOCO01 status={loco_status}, decision={loco_decision}."},
        {"claim_id": "C06", "claim": "HyDRA-VLM improves centre-level safety or reduces CIN3+ FN.", "classification": "NOT_VERIFIED", "support": "LOCO01", "safe_wording": "do not claim for HyDRA-VLM without true model", "notes": "Cached-adapter run is not full HyDRA-VLM."},
        {"claim_id": "C07", "claim": "Reliability gating reduces centre gap.", "classification": "SUPPLEMENT_ONLY" if abl01_status == "PASS" else "NOT_VERIFIED", "support": "ABL01", "safe_wording": "lightweight cached-feature reliability proxy", "notes": "Not a full trainable HyDRA-VLM module claim."},
        {"claim_id": "C08", "claim": "Iterative evidence accumulation reduces CIN3+ FN.", "classification": "SUPPLEMENT_ONLY" if abl01_status == "PASS" else "NOT_VERIFIED", "support": "ABL01", "safe_wording": "proxy evidence-accumulation comparison", "notes": "Use only with caveat."},
        {"claim_id": "C09", "claim": "Guideline alignment improves calibration/ECE.", "classification": "NOT_VERIFIED", "support": "ABL01", "safe_wording": "[TO VERIFY]", "notes": "No trainable guideline/prototype loss was implemented in recovery."},
        {"claim_id": "C10", "claim": "Score-level TTA reduces CIN3+ FN.", "classification": "ALLOWED_WITH_CAVEAT" if routeb_tta else "NOT_VERIFIED", "support": "Route B Table_TTA_Comparison_IF", "safe_wording": "transductive score-level TTA reduced CIN3+ FN", "notes": "Not source-only deployment."},
        {"claim_id": "C11", "claim": "Score-level TTA does not materially improve AUC.", "classification": "ALLOWED_WITH_CAVEAT" if routeb_tta else "NOT_VERIFIED", "support": "Route B TTA/ROC outputs", "safe_wording": "TTA shifted operating points without materially improving AUC", "notes": ""},
        {"claim_id": "C12", "claim": "Hard-centre failure is ranking-level.", "classification": "ALLOWED_WITH_CAVEAT" if routeb_centre else "NOT_VERIFIED", "support": "Route B hard-centre analysis", "safe_wording": "hard-centre ranking-level repairability boundary", "notes": "Do not say rescued."},
        {"claim_id": "C13", "claim": "CoE is a transparency aid.", "classification": "ALLOWED_WITH_CAVEAT" if coe_status == "PASS" else "ALLOWED_WITH_CAVEAT", "support": "COE01", "safe_wording": "CoE is a transparency aid pending expert validation", "notes": ""},
        {"claim_id": "C14", "claim": "CoE is clinically faithful.", "classification": "PROHIBITED", "support": "none", "safe_wording": "pending expert validation", "notes": "No expert validation."},
        {"claim_id": "C15", "claim": "HyDRA-VLM is deployment-ready.", "classification": "PROHIBITED", "support": "none", "safe_wording": "requires prospective validation", "notes": ""},
        {"claim_id": "C16", "claim": "HyDRA-VLM is SOTA.", "classification": "PROHIBITED", "support": "none", "safe_wording": "selected locked-protocol baselines only", "notes": ""},
    ]
    df = pd.DataFrame(rows)
    C.write_csv(OUT / "R01_HyDRA_VLM_Updated_Claim_Lock.csv", df)
    ready = "NO"
    missing = [
        "Auditable fold-wise BioMedCLIP-LoRA training/evaluation outputs.",
        "True HyDRA-VLM locked five-centre LOCO metrics.",
        "Module ablations for full trainable HyDRA-VLM.",
        "VLM-LoRA MMD reduction evidence.",
        "Guideline-alignment calibration/ECE ablation.",
    ]
    if lora_verified and loco_decision in {"YES", "YES_WITH_CAVEAT_CACHED_ADAPTER_NOT_LORA"}:
        ready = "YES_WITH_CAVEATS"
    report = [
        "# R01 HyDRA-VLM Updated Claim Lock",
        "",
        C.md_table(df),
    ]
    C.write_text(OUT / "R01_HyDRA_VLM_Updated_Claim_Lock.md", "\n".join(report) + "\n")
    decision = [
        "# R01 Manuscript Readiness Decision",
        "",
        f"READY_FOR_M02_M12 = {ready}",
        "",
        "Reason: recovery outputs completed, but BioMedCLIP-LoRA and true HyDRA-VLM LOCO claims remain not verified. Manuscript should remain Route B unless these missing experiments are completed.",
        "",
        "## Remaining Missing Outputs",
        "",
    ]
    decision += [f"- {m}" for m in missing]
    C.write_text(OUT / "R01_Manuscript_Readiness_Decision.md", "\n".join(decision) + "\n")
    C.status_json(OUT / "status.json", "PASS", "Claim lock updated.", READY_FOR_M02_M12=ready)
    C.file_manifest(OUT, OUT / "r01_file_manifest.csv")


if __name__ == "__main__":
    main()
