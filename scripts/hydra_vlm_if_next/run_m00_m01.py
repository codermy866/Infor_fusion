#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_revisions" / "hydra_vlm_if"
SECTIONS = OUT / "sections"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
AUDIT = OUT / "audit"
PROMPTS = OUT / "prompts"


@dataclass
class InventoryRow:
    file_path: str
    file_type: str
    exists: bool
    size_bytes: int
    last_modified: str
    role: str
    usable_for_revision: bool
    notes: str


def ensure_dirs() -> None:
    for d in [OUT, SECTIONS, TABLES, FIGURES, AUDIT, PROMPTS]:
        d.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_info(path: Path, role: str = "UNKNOWN", usable: bool = False, notes: str = "") -> InventoryRow:
    if path.exists():
        st = path.stat()
        modified = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        size = st.st_size
        suffix = path.suffix.lower().lstrip(".") or ("dir" if path.is_dir() else "unknown")
        return InventoryRow(rel(path), suffix, True, size, modified, role, usable, notes)
    return InventoryRow(rel(path), path.suffix.lower().lstrip(".") or "missing", False, 0, "", role, False, notes or "Expected candidate not found.")


def classify(path: Path) -> tuple[str, bool, str]:
    p = rel(path).lower()
    name = path.name.lower()
    ext = path.suffix.lower()

    role = "UNKNOWN"
    usable = False
    notes = ""

    if ext in {".md", ".tex"} and (
        "manuscript" in p
        or "sections" in p
        or "if_route_b_submission_pack" in p
        or name in {"main.tex", "information_fusion.tex", "paper.md", "manuscript.md"}
    ):
        role = "MANUSCRIPT_SOURCE"
        usable = True
    if "if_route_b_submission_pack/supplement" in p:
        role = "SUPPLEMENT"
        usable = True
    if ext in {".csv", ".tex"} and ("table" in name or "/tables/" in p or "source_csv" in p):
        role = "FINAL_TABLE"
        usable = True
    if ext in {".png", ".pdf", ".svg"} and ("figure" in name or "/figures/" in p):
        role = "FINAL_FIGURE"
        usable = True

    if "step2_5_full_hydra_vlm_recovery" in p or "vlm" in p or "lora" in p or "biomedclip" in p:
        role = "VLM_OUTPUT"
        usable = True
        notes = "VLM-related artifact; verify whether it is fold-wise BioMedCLIP-LoRA before using method claims."
    if "loco" in p or "step2_main_loco" in p or "if_route_b_master" in p or "if_route_b_submission_pack" in p:
        role = "LOCO_OUTPUT"
        usable = True
    if "ablation" in p or "abl_" in p or "table_ablation" in p:
        role = "ABLATION_OUTPUT"
        usable = True
    if "domain" in p or "mmd" in p or "umap" in p or "step2_9" in p or "centre_classifier" in p:
        role = "DOMAIN_SHIFT_OUTPUT"
        usable = True
    if "failure" in p or "hardcentre" in p or "hard_centre" in p or "xiangyang" in p or "fn_" in p:
        role = "FAILURE_ANALYSIS_OUTPUT"
        usable = True
    if "tta" in p or "step2_10" in p or "target_adaptation" in p:
        role = "TTA_OUTPUT"
        usable = True
    if "clinical" in p or "dca" in p or "ece" in p or "calibration" in p:
        role = "CLINICAL_OUTPUT"
        usable = True
    if "coe" in p or "chain" in p:
        role = "COE_OUTPUT"
        usable = True

    if ".routeb_python_pkgs" in p or "__pycache__" in p:
        usable = False
        notes = "Local/generated dependency or cache; not a manuscript evidence source."

    return role, usable, notes


def add_existing(rows: list[InventoryRow], path: Path, role: str | None = None, notes: str = "") -> None:
    if role is None:
        role, usable, auto_notes = classify(path)
        rows.append(file_info(path, role, usable, notes or auto_notes))
    else:
        rows.append(file_info(path, role, path.exists(), notes))


def iter_candidate_files() -> list[Path]:
    candidates: list[Path] = []
    explicit = [
        ROOT / "outputs/publishable_v2/if_route_b_submission_pack/IF_RouteB_Manuscript_Draft.md",
        ROOT / "outputs/publishable_v2/if_route_b_submission_pack/IF_RouteB_Supplementary_Material.md",
        ROOT / "outputs/publishable_v2/if_route_b_submission_pack/Algorithm_RouteB_Locked_LOCO_Pseudocode.tex",
        ROOT / "outputs/publishable_v2/if_route_b_master/IF_RouteB_Master_Execution_Report.md",
        ROOT / "outputs/publishable_v2/if_route_b_remaining/IF_RouteB_Remaining_Execution_Report.md",
        ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery/STEP2_5_FULL_HYDRA_VLM_RECOVERY_STATUS.md",
        ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery/STEP2_9_DG_RECOVERY_STATUS.md",
        ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision/STEP2_10_TARGET_ADAPTATION_IF_STATUS.md",
        ROOT / "Final_IF_Experiment_Plan.md",
        ROOT / "Information_Fusion.tex",
        ROOT / "main.tex",
        ROOT / "manuscript.md",
        ROOT / "paper.md",
    ]
    candidates.extend(explicit)

    roots = [
        ROOT / "outputs/publishable_v2/if_route_b_submission_pack",
        ROOT / "outputs/publishable_v2/if_route_b_master",
        ROOT / "outputs/publishable_v2/if_route_b_remaining",
        ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery",
        ROOT / "outputs/publishable_v2/step2_8_auc_recovery_information_fusion",
        ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery",
        ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision",
        ROOT / "outputs/publishable_v2/step2_main_loco",
        ROOT / "paper_revision/final_outputs/infofusion_revision_package",
        ROOT / "paper_revision/results",
        ROOT / "paper_revision/tables",
        ROOT / "paper_revision/figures",
        ROOT / "paper_revision/manuscript_sections",
        ROOT / "final_result",
    ]
    allowed_ext = {".md", ".tex", ".csv", ".png", ".pdf", ".svg", ".json"}
    for base in roots:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "checkpoints", "cache"}]
            for fn in filenames:
                path = Path(dirpath) / fn
                if path.suffix.lower() in allowed_ext:
                    candidates.append(path)

    expected_dirs = [
        ROOT / "experiments/vlm_finetune",
        ROOT / "experiments/loco_hydra_vlm",
        ROOT / "experiments/ablation",
        ROOT / "experiments/domain_shift",
        ROOT / "experiments/failure_analysis",
        ROOT / "experiments/tta",
        ROOT / "experiments/clinical",
        ROOT / "experiments/coe",
        ROOT / "if_route_b_submission_pack",
        ROOT / "paper_tables",
        ROOT / "paper_figures",
    ]
    candidates.extend(expected_dirs)

    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = [str(row.get(c, "")).replace("\n", " ") for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def read_text(path: Path) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_support(rows: list[InventoryRow]) -> dict[str, bool | str]:
    paths = [r.file_path for r in rows if r.exists]
    joined = "\n".join(paths).lower()
    step25_status = read_text(ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery/STEP2_5_FULL_HYDRA_VLM_RECOVERY_STATUS.md").lower()
    route_b_tta = read_text(ROOT / "outputs/publishable_v2/if_route_b_master/tables/Table_TTA_Comparison_IF.csv")
    centre = read_text(ROOT / "outputs/publishable_v2/if_route_b_master/tables/Table_Centre_Level_Results_IF.csv")
    final_plan_exists = (ROOT / "Final_IF_Experiment_Plan.md").exists()

    vlm_lora_paths = [p for p in paths if "lora" in p.lower() and ("metric" in p.lower() or "result" in p.lower() or "loco" in p.lower())]
    has_biomedclip_lora = bool(vlm_lora_paths) and "failed_partial" not in step25_status
    has_hydra_vlm_loco = (
        "experiments/loco_hydra_vlm" in joined
        or ("hydra-vlm" in joined and "aggregate" in joined)
        or ("full hydra-coe + auxiliary oct ssl + oct-vlm alignment" in step25_status and "not_run" not in step25_status)
    )
    has_route_b = (ROOT / "outputs/publishable_v2/if_route_b_master/tables/Table_TTA_Comparison_IF.csv").exists()
    tta_fn_supported = "0.906,18" in route_b_tta and "0.942,11" in route_b_tta
    tta_auc_boundary_supported = "0.741" in route_b_tta and "confidence-filtered" in route_b_tta.lower()
    hard_centre_supported = "hardest-centre" in centre.lower() and "ranking not repaired" in centre.lower()

    return {
        "final_plan_exists": final_plan_exists,
        "has_biomedclip_lora": has_biomedclip_lora,
        "has_hydra_vlm_loco": has_hydra_vlm_loco,
        "has_ablation_outputs": any("ablation" in p.lower() for p in paths),
        "has_route_b": has_route_b,
        "tta_fn_supported": tta_fn_supported,
        "tta_auc_boundary_supported": tta_auc_boundary_supported,
        "hard_centre_supported": hard_centre_supported,
        "step25_status": "FAILED_PARTIAL_RECOVERY_ONLY" if "failed_partial_recovery_only" in step25_status else "UNKNOWN_OR_NOT_FOUND",
    }


def run_m00() -> dict[str, bool | str]:
    rows: list[InventoryRow] = []
    for path in iter_candidate_files():
        role, usable, notes = classify(path)
        rows.append(file_info(path, role, usable and path.exists(), notes))

    inv_rows = [r.__dict__ for r in rows]
    write_csv(AUDIT / "M00_file_inventory.csv", inv_rows)
    support = detect_support(rows)

    role_counts: dict[str, int] = {}
    for r in rows:
        if r.exists:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

    missing = [
        "Final_IF_Experiment_Plan.md was not found in the current repository root.",
        "Fold-wise BioMedCLIP-LoRA training/evaluation outputs for the locked five-centre LOCO protocol.",
        "HyDRA-VLM aggregate_metrics.csv and centre_level_metrics.csv under an auditable loco_hydra_vlm experiment directory.",
        "ABL-01 module-level ablation for VLM encoding, reliability gating, iterative evidence accumulation, and guideline alignment under locked LOCO.",
        "ABL-04 VLM backbone ablation comparing ViT+BioBERT, frozen BioMedCLIP, and BioMedCLIP-LoRA.",
        "Verified VLM-LoRA centre-shift/MMD reduction outputs.",
        "VLM decoder/CoE quantitative or expert-validation outputs.",
    ]
    report = [
        "# M00 Manuscript and Output Inventory",
        "",
        "## Decision Summary",
        "",
        "- Recommended manuscript source to edit: `outputs/publishable_v2/if_route_b_submission_pack/IF_RouteB_Manuscript_Draft.md` as the Route B source draft; write HyDRA-VLM revision material into `paper_revisions/hydra_vlm_if/sections/`.",
        f"- VLM-LoRA outputs exist: `{bool(support['has_biomedclip_lora'])}`.",
        f"- LOCO HyDRA-VLM outputs exist: `{bool(support['has_hydra_vlm_loco'])}`.",
        f"- Ablation outputs exist: `{bool(support['has_ablation_outputs'])}`; however, current detected ablations are historical/proxy or Route-B supportive, not verified HyDRA-VLM VLM-LoRA module ablations.",
        "- Current paper can be rewritten now: `PARTIAL_ONLY`. Per the prompt rule, only M00-M01 should be executed before VLM-LoRA/HyDRA-VLM LOCO outputs are generated.",
        f"- Step2.5 VLM status: `{support['step25_status']}`.",
        "",
        "## Existing Output Counts By Role",
        "",
        md_table([{"role": k, "existing_files": v} for k, v in sorted(role_counts.items())], ["role", "existing_files"]),
        "",
        "## Missing Outputs Blocking Full M02-M12 Rewrite",
        "",
    ]
    report += [f"- {m}" for m in missing]
    report += [
        "",
        "## Notes",
        "",
        "The current Route B package supports the locked n=1897 five-centre LOCO benchmark, source-only versus score-level TTA separation, TTA false-negative reduction, and hard-centre ranking-boundary analysis. It does not yet verify the upgraded HyDRA-VLM claims about fold-wise BioMedCLIP-LoRA, VLM-LoRA feature-quality gains, VLM-driven MMD reduction, or HyDRA-VLM module ablations.",
    ]
    (AUDIT / "M00_inventory_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (AUDIT / "M00_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return support


def claim_rows(support: dict[str, bool | str]) -> list[dict]:
    route_b_source = "outputs/publishable_v2/if_route_b_master; outputs/publishable_v2/if_route_b_submission_pack"
    vlm_required = "experiments/vlm_finetune or experiments/loco_hydra_vlm fold-wise BioMedCLIP-LoRA metrics"
    return [
        {
            "claim_id": "C01",
            "claim_text": "Locked n=1897 five-centre LOCO benchmark.",
            "claim_category": "CAVEATED",
            "supporting_required_output": route_b_source,
            "current_support_status": "SUPPORTED_BY_ROUTE_B_OUTPUTS",
            "allowed_wording": "We evaluated the framework under a locked n=1897 five-centre LOCO benchmark.",
            "forbidden_wording": "The benchmark proves deployment safety.",
            "manuscript_section": "Methods; Results",
            "notes": "Supported as benchmark/evaluation claim.",
        },
        {
            "claim_id": "C02",
            "claim_text": "VLM-LoRA feature extraction was performed fold-wise without target-label leakage.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": vlm_required,
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] Fold-wise BioMedCLIP-LoRA was trained on source centres only.",
            "forbidden_wording": "VLM-LoRA was leakage-free without fold-wise logs.",
            "manuscript_section": "Methods",
            "notes": "No auditable fold-wise BioMedCLIP-LoRA outputs detected.",
        },
        {
            "claim_id": "C03",
            "claim_text": "VLM-LoRA improves feature quality over frozen BioMedCLIP.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "ABL-04 VLM backbone ablation with frozen BioMedCLIP and BioMedCLIP-LoRA.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] VLM-LoRA improved feature quality relative to frozen BioMedCLIP.",
            "forbidden_wording": "VLM-LoRA improves AUC/feature quality without completed ablation.",
            "manuscript_section": "Results",
            "notes": "Step2.5 reports compute-limited OCT-VLM proxy, not full BioMedCLIP-LoRA evidence.",
        },
        {
            "claim_id": "C04",
            "claim_text": "VLM-LoRA reduces centre-shift MMD.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "Pre/post VLM-LoRA MMD table and figure.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] VLM-LoRA reduced centre-shift MMD in frozen patient-level feature space.",
            "forbidden_wording": "VLM-LoRA reduces centre shift based on Route B MMD alone.",
            "manuscript_section": "Results",
            "notes": "Route B MMD is descriptive, not a verified VLM-LoRA delta.",
        },
        {
            "claim_id": "C05",
            "claim_text": "HyDRA-VLM improves pooled CIN2+ AUC over the previous HyDRA-DG baseline.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "HyDRA-VLM locked LOCO aggregate metrics versus HyDRA-DG source-only baseline.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] HyDRA-VLM improved pooled CIN2+ AUC over HyDRA-DG.",
            "forbidden_wording": "HyDRA-VLM improves pooled AUC without completed locked LOCO run.",
            "manuscript_section": "Results",
            "notes": "The current strongest verified source-only Route B row is HyDRA-DG, not HyDRA-VLM.",
        },
        {
            "claim_id": "C06",
            "claim_text": "Reliability gating reduces centre gap.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "Module ablation comparing full model versus no reliability gating under locked LOCO.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] Reliability-gated fusion reduced centre-level performance gap.",
            "forbidden_wording": "Reliability gating makes the representation centre-invariant.",
            "manuscript_section": "Results",
            "notes": "Needs ABL-01 or equivalent module ablation.",
        },
        {
            "claim_id": "C07",
            "claim_text": "Iterative evidence accumulation reduces CIN3+ false negatives.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "Module ablation comparing full model versus no iterative evidence accumulation.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] Iterative evidence accumulation reduced CIN3+ false negatives.",
            "forbidden_wording": "Evidence accumulation improves safety without CIN3+ FN ablation.",
            "manuscript_section": "Results",
            "notes": "Needs locked LOCO module ablation.",
        },
        {
            "claim_id": "C08",
            "claim_text": "Guideline alignment improves calibration/ECE.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "Module C ablation with ECE/calibration metrics.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] Guideline alignment improved calibration.",
            "forbidden_wording": "Guideline alignment is clinically validated.",
            "manuscript_section": "Results",
            "notes": "Route B ECE exists, but improvement from Module C is not verified.",
        },
        {
            "claim_id": "C09",
            "claim_text": "Score-level TTA reduces CIN3+ FN.",
            "claim_category": "CAVEATED",
            "supporting_required_output": "outputs/publishable_v2/if_route_b_master/tables/Table_TTA_Comparison_IF.csv",
            "current_support_status": "SUPPORTED_WITH_TRANSDUCTIVE_CAVEAT" if support["tta_fn_supported"] else "NOT_VERIFIED",
            "allowed_wording": "In a transductive score-level analysis, the best TTA candidate reduced CIN3+ false negatives.",
            "forbidden_wording": "TTA improves source-only deployment safety.",
            "manuscript_section": "Results; Discussion",
            "notes": "Current Route B table reports source-only 18 FN and best score-level TTA 11 FN.",
        },
        {
            "claim_id": "C10",
            "claim_text": "Score-level TTA does not materially improve AUC.",
            "claim_category": "CAVEATED",
            "supporting_required_output": "Table_TTA_Comparison_IF.csv; Figure_ROC_TTA_Analysis",
            "current_support_status": "SUPPORTED" if support["tta_auc_boundary_supported"] else "NOT_VERIFIED",
            "allowed_wording": "Score-level TTA shifted operating points but did not materially improve pooled AUC.",
            "forbidden_wording": "Score-level TTA repairs ranking failure.",
            "manuscript_section": "Results; Discussion",
            "notes": "Current Route B table reports source-only and best TTA CIN2+ AUC both 0.741.",
        },
        {
            "claim_id": "C11",
            "claim_text": "Hard-centre failure is ranking-level rather than calibration-level.",
            "claim_category": "CAVEATED",
            "supporting_required_output": "Figure_Ranking_vs_Calibration_Diagnosis; hard-centre tables.",
            "current_support_status": "SUPPORTED_WITH_CAVEAT" if support["hard_centre_supported"] else "NOT_VERIFIED",
            "allowed_wording": "The hardest centre showed a ranking-level repairability boundary rather than a pure calibration failure.",
            "forbidden_wording": "Xiangyang was rescued.",
            "manuscript_section": "Results; Discussion",
            "notes": "Supported by Route B hard-centre analysis, but needs cautious language.",
        },
        {
            "claim_id": "C12",
            "claim_text": "CoE is improved by VLM decoder.",
            "claim_category": "ALLOWED_IF_VERIFIED",
            "supporting_required_output": "COE-01/02/03 VLM decoder outputs and proxy/expert metrics.",
            "current_support_status": "NOT_VERIFIED",
            "allowed_wording": "[TO VERIFY] VLM decoder improved CoE proxy metrics.",
            "forbidden_wording": "VLM decoder provides clinically faithful explanations.",
            "manuscript_section": "Results; Discussion",
            "notes": "Current CoE remains a transparency aid pending expert validation.",
        },
        {
            "claim_id": "C13",
            "claim_text": "CoE is clinically faithful.",
            "claim_category": "PROHIBITED",
            "supporting_required_output": "Independent expert faithfulness validation.",
            "current_support_status": "NO_INDEPENDENT_EVIDENCE",
            "allowed_wording": "CoE is a transparency aid pending expert validation.",
            "forbidden_wording": "Clinically faithful CoE; clinically validated reasoning.",
            "manuscript_section": "None",
            "notes": "Explicitly prohibited unless independent expert evidence is added.",
        },
        {
            "claim_id": "C14",
            "claim_text": "HyDRA-VLM is deployment-ready.",
            "claim_category": "PROHIBITED",
            "supporting_required_output": "Prospective deployment validation and clinical safety protocol.",
            "current_support_status": "NO_PROSPECTIVE_EVIDENCE",
            "allowed_wording": "Future work requires prospective locked-threshold validation.",
            "forbidden_wording": "Deployment-ready; safe clinical deployment.",
            "manuscript_section": "None",
            "notes": "Prohibited.",
        },
        {
            "claim_id": "C15",
            "claim_text": "HyDRA-VLM is state-of-the-art across all benchmarks.",
            "claim_category": "PROHIBITED",
            "supporting_required_output": "Broad external benchmark comparisons.",
            "current_support_status": "NO_BROAD_BENCHMARK_EVIDENCE",
            "allowed_wording": "Compared with selected locked-protocol baselines.",
            "forbidden_wording": "State-of-the-art across all benchmarks; outperforms all baselines.",
            "manuscript_section": "None",
            "notes": "Prohibited.",
        },
    ]


def run_m01(support: dict[str, bool | str]) -> None:
    rows = claim_rows(support)
    write_csv(AUDIT / "M01_HyDRA_VLM_Claim_Lock.csv", rows)
    columns = [
        "claim_id",
        "claim_text",
        "claim_category",
        "current_support_status",
        "allowed_wording",
        "forbidden_wording",
    ]
    positioning = (
        "HyDRA-VLM is presented as a reliability-aware multimodal fusion framework evaluated under a locked multicentre LOCO protocol. "
        "VLM-enhanced encoding and reliability-gated fusion are evaluated as method components, while score-level TTA is analysed as a repairability probe rather than a deployment strategy."
    )
    report = [
        "# M01 HyDRA-VLM Claim-Lock Checklist",
        "",
        "## Execution Decision",
        "",
        "Per the prompt rule, full M02-M12 manuscript rewriting is paused because fold-wise BioMedCLIP-LoRA and HyDRA-VLM locked-LOCO outputs are not yet verified.",
        "",
        "## Manuscript-Safe Positioning",
        "",
        positioning,
        "",
        "## Claim Table",
        "",
        md_table(rows, columns),
        "",
        "## Blocking Missing Evidence",
        "",
        "- Fold-wise BioMedCLIP-LoRA logs/checkpoints/metrics without target-label leakage.",
        "- Locked five-centre LOCO HyDRA-VLM aggregate and centre-level metrics.",
        "- ABL-01 module ablation for VLM encoding, reliability gating, iterative evidence accumulation, and guideline alignment.",
        "- ABL-04 VLM backbone ablation.",
        "- VLM-LoRA MMD reduction analysis.",
        "- VLM decoder/CoE validation outputs if CoE improvement is claimed.",
    ]
    text = "\n".join(report) + "\n"
    (AUDIT / "M01_HyDRA_VLM_Claim_Lock.md").write_text(text, encoding="utf-8")
    (AUDIT / "M01_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    support = run_m00()
    run_m01(support)
    summary_rows = [
        {"operation_id": "M00", "status": "COMPLETED", "report": rel(AUDIT / "M00_inventory_report.md")},
        {"operation_id": "M01", "status": "COMPLETED", "report": rel(AUDIT / "M01_HyDRA_VLM_Claim_Lock.md")},
        {
            "operation_id": "M02-M12",
            "status": "PAUSED_BY_PROMPT_RULE",
            "report": "Missing fold-wise BioMedCLIP-LoRA and HyDRA-VLM locked-LOCO outputs.",
        },
    ]
    write_csv(AUDIT / "M00_M01_execution_summary.csv", summary_rows)
    print("M00-M01 completed. M02-M12 paused because VLM-LoRA/HyDRA-VLM LOCO outputs are not verified.")
    print(rel(AUDIT / "M00_inventory_report.md"))
    print(rel(AUDIT / "M01_HyDRA_VLM_Claim_Lock.md"))


if __name__ == "__main__":
    main()
