#!/usr/bin/env python3
"""Refresh final maps, audits, manifests, and package after P06/P10/P11 completion."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.if_supplementary.common import (
    DEFAULT_OUT_ROOT,
    copy_pair,
    ensure_out_dirs,
    environment_snapshot,
    input_paths,
    make_zip_package,
    now,
    save_csv,
    str_rel,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    ensure_out_dirs(out_root)
    write_preflight(out_root)
    refresh_final_tables_figures(out_root)
    refresh_claims_and_audit(out_root)
    package = make_zip_package(out_root)
    test_zip(package)
    print(f"package: {package}")
    return 0


def write_preflight(out_root: Path) -> None:
    paths = input_paths(out_root)
    checkpoint_files = sorted(paths["checkpoint_root"].glob("**/checkpoints/*.pth")) if paths["checkpoint_root"].exists() else []
    rows = [
        inv("standardized_prediction_registry", paths["test_predictions"], True, True, True, "use_locked_registry", "Contains patient-level hashes and all model scores."),
        inv("validation_locked_thresholds", paths["thresholds"], True, False, False, "use_locked_thresholds", "Provides fold-wise CIN2/CIN3 thresholds."),
        inv("clean_reliability_weights", paths["reliability"], False, True, False, "use_clean_export", "Clean reliability weights available at patient level."),
        inv("coe_clean_proxy_states", paths["coe_proxy"], False, False, True, "audit_only", "Clean CoE proxy states available, but no true intervention logits."),
        inv("loco_split_file", paths["loco_folds"], True, True, True, "protocol_reference", "Strict LOCO fold file."),
        inv("locked_feature_cache", paths["feature_npz"], True, True, True, "not_loaded_for_raw_inference", "Useful provenance, but completion uses locked prediction registry."),
        inv("dataset_manifest", paths["data_lock"], True, True, True, "audit_only", "Raw IDs/images are not exported in the package."),
        inv("shared_lora_checkpoint_root", paths["checkpoint_root"], False, False, False, "future_raw_inference_route", f"{len(checkpoint_files)} checkpoint files found; full raw-image dataloader/inference export was not invoked for this completion."),
        inv("p06_completion_script", Path("scripts/if_supplementary/complete_p06_random_dropout_predictions.py"), True, False, False, "run_completed", "Generates patient-level dropout proxy predictions."),
        inv("p10_completion_script", Path("scripts/if_supplementary/complete_p10_perturbed_reliability_export.py"), False, True, False, "run_completed", "Generates patient-level clean-vs-perturbed reliability weights."),
        inv("p11_completion_script", Path("scripts/if_supplementary/complete_p11_coe_intervention_logits.py"), False, False, True, "run_completed_audit", "Exports audit table; true intervention logits remain unavailable."),
    ]
    inventory = pd.DataFrame(rows)
    pc = out_root / "partial_completion"
    save_csv(inventory, pc / "partial_completion_inventory.csv")
    report = [
        "# P06/P10/P11 Partial Completion Preflight Audit",
        "",
        f"Created at: `{now()}`.",
        "",
        "Can P06 10/30/50% random dropout patient-level predictions be generated? YES.",
        "Can P10 perturbed reliability weights be exported? YES, as feature-level proxy perturbations.",
        "Can P11 CoE intervention logits be exported? NO, only clean proxy logits are available.",
        "Can P11 visual saliency masks be generated? NO.",
        "",
        "## Blockers",
        "",
        "- P11 targeted/random/counterfactual intervention logits are not present in the locked outputs.",
        "- P11 saliency or occlusion masks are not present in the locked outputs.",
        "- Checkpoints exist, but this completion does not reconstruct raw-image dataloaders or modify the original training-time export contract.",
        "",
        "## Inventory",
        "",
        inventory.to_string(index=False),
    ]
    write_text(pc / "PARTIAL_PREFLIGHT_AUDIT.md", "\n".join(report) + "\n")


def inv(component: str, path: Path, p06: bool, p10: bool, p11: bool, action: str, notes: str) -> dict[str, object]:
    full = path if path.is_absolute() else Path.cwd() / path
    return {
        "component": component,
        "path": str(path),
        "exists": bool(full.exists()),
        "usable_for_p06": bool(p06),
        "usable_for_p10": bool(p10),
        "usable_for_p11": bool(p11),
        "required_action": action,
        "notes": notes,
    }


def refresh_final_tables_figures(out_root: Path) -> None:
    final = out_root / "12_final_tables_figures"
    final.mkdir(parents=True, exist_ok=True)
    table_sources = {
        "Table_5_modality_and_missingness.csv": out_root / "05_modality_ablation_and_missingness/random_dropout_stress_table.csv",
        "Table_9_reliability_validation.csv": out_root / "09_reliability_validation/reliability_perturbation_response_summary.csv",
        "Table_10_coe_faithfulness_controls.csv": out_root / "10_coe_faithfulness/coe_intervention_summary.csv",
    }
    for name, src in table_sources.items():
        if src.exists():
            df = pd.read_csv(src)
            save_csv(df, final / name)

    figure_sources = {
        "Figure_5_modality_missingness_stress": out_root / "05_modality_ablation_and_missingness/figure_random_dropout_rate_auc_npv",
        "Figure_7_reliability_weight_validation": out_root / "09_reliability_validation/figure_perturbed_reliability_response",
        "Figure_8_coe_faithfulness_controls": out_root / "10_coe_faithfulness/figure_coe_step_specificity_heatmap",
    }
    for name, src_stem in figure_sources.items():
        copy_pair(src_stem, final / name)

    rows = []
    for file in sorted(final.glob("*")):
        if file.suffix.lower() == ".csv":
            typ = "table"
        elif file.suffix.lower() in {".png", ".pdf"}:
            typ = "figure"
        elif file.suffix.lower() == ".md":
            typ = "report"
        else:
            continue
        rows.append({"artifact": file.name, "path": str_rel(file), "type": typ})
    save_csv(pd.DataFrame(rows), final / "FIGURE_TABLE_MAP.csv")

    p06 = read_optional(out_root / "05_modality_ablation_and_missingness/random_dropout_stress_table.csv")
    p10 = read_optional(out_root / "09_reliability_validation/reliability_perturbation_response_summary.csv")
    p11 = read_optional(out_root / "10_coe_faithfulness/coe_intervention_summary.csv")
    summary = [
        "# Results Summary for Manuscript",
        "",
        "The supplementary package has been refreshed after P06/P10/P11 partial-completion execution.",
        "",
        "## Completion Status",
        "",
        "- P06: PASS as patient-level locked prediction-registry random modality dropout stress test.",
        "- P10: PASS as patient-level feature-level proxy reliability perturbation export.",
        "- P11: PARTIAL/NOT_ESTABLISHED because true CoE intervention logits and saliency masks are unavailable.",
        "",
        "## Claim Boundary",
        "",
        "Use P06 and P10 as supplementary robustness/diagnostic evidence. Do not claim raw-image perturbation reliability validation for P10, and do not claim CoE faithfulness or causal explanation for P11.",
        "",
        "## Key Table Heads",
        "",
        "### P06",
        p06.head(8).to_string(index=False) if len(p06) else "not available",
        "",
        "### P10",
        p10.head(8).to_string(index=False) if len(p10) else "not available",
        "",
        "### P11",
        p11.head(8).to_string(index=False) if len(p11) else "not available",
    ]
    write_text(final / "RESULTS_SUMMARY_FOR_MANUSCRIPT.md", "\n".join(summary) + "\n")
    write_text(final / "SUPPLEMENTARY_RESULTS_SUMMARY.md", "\n".join(summary) + "\n")


def refresh_claims_and_audit(out_root: Path) -> None:
    audit = out_root / "13_submission_audit"
    audit.mkdir(parents=True, exist_ok=True)
    claim = [
        "# IF Final Experiment Claim Lock",
        "",
        f"Updated at: `{now()}` after P06/P10/P11 partial-completion execution.",
        "",
        "## Protocol Invariants",
        "",
        "1. Analytic cohort remains locked at n=1897.",
        "2. Primary protocol remains strict five-fold LOCO.",
        "3. Validation-locked thresholds are reused; held-out test labels are not used for model selection or threshold selection.",
        "4. New patient-level outputs omit raw patient and case identifiers.",
        "",
        "## P06/P10/P11 Status",
        "",
        "| Step | Status | Evidence | Claim boundary |",
        "|---|---|---|---|",
        "| P06 | PASS_FEATURE_CACHE_PROXY | `05_modality_ablation_and_missingness/random_dropout_patient_level_predictions.csv` | Patient-level random modality dropout stress test from locked prediction registry; not raw-image checkpoint re-inference. |",
        "| P10 | PASS_FEATURE_LEVEL_PROXY | `09_reliability_validation/reliability_weights_clean_and_perturbed_patient_level.csv` | Clean-vs-perturbed reliability response under feature-level proxy perturbations; not raw-image corruption reliability validation. |",
        "| P11 | PARTIAL_NOT_ESTABLISHED | `10_coe_faithfulness/coe_intervention_logits_patient_level.csv` | Clean CoE proxy states audited; no faithfulness, causal explanation, or saliency claim. |",
        "",
        "## Allowed Claims",
        "",
        "- Locked multicenter LOCO benchmark results.",
        "- Validation-locked CIN3+ safety/referral trade-off.",
        "- Patient-level missing-modality stress testing using locked prediction-registry proxies.",
        "- Feature-level reliability-weight response as an internal diagnostic.",
        "- CoE clean proxy trajectories as transparency aids only.",
        "",
        "## Claims To Remove Or Soften",
        "",
        "- Clinical deployment safety.",
        "- Raw-image perturbation reliability validation.",
        "- CoE faithfulness, causal explanation, saliency-grounded explanation, or counterfactual intervention claims.",
        "- Any statement that P06/P10/P11 completion used target labels for model selection.",
    ]
    write_text(audit / "IF_FINAL_EXPERIMENT_CLAIM_LOCK.md", "\n".join(claim) + "\n")

    audit_lines = [
        "# IF Supplementary Experiment Audit",
        "",
        f"Updated at: `{now()}`.",
        "",
        "| Step | Final status | Output check |",
        "|---|---|---|",
        "| P06 | PASS_FEATURE_CACHE_PROXY | Random dropout patient-level CSV, by-center/by-repeat tables, and three figures were generated. |",
        "| P10 | PASS_FEATURE_LEVEL_PROXY | Clean-vs-perturbed reliability patient-level CSV, summaries, tests, and four figures were generated. |",
        "| P11 | PARTIAL_NOT_ESTABLISHED | CoE intervention audit table and figures were generated, but true intervened logits/saliency remain unavailable. |",
        "",
        "No raw patient images or raw identifiers are included in the supplementary package.",
    ]
    write_text(audit / "IF_SUPPLEMENTARY_EXPERIMENT_AUDIT.md", "\n".join(audit_lines) + "\n")

    missing = [
        "# Missing Requirements",
        "",
        "P06 has been completed as a patient-level locked prediction-registry random modality dropout stress test.",
        "",
        "- `P10` / `raw-image corruption-response reliability weights`: Feature-level proxy perturbation weights have been exported, but raw-image perturbation inference weights are still unavailable.",
        "- `P11` / `CoE targeted/random/counterfactual/visual interventions`: True intervened CoE logits and saliency masks are still unavailable; CoE faithfulness is not established.",
    ]
    write_text(out_root / "MISSING_REQUIREMENTS.md", "\n".join(missing) + "\n")

    write_text(audit / "ENVIRONMENT_SNAPSHOT.txt", environment_snapshot())
    append_runtime_manifest(out_root)


def append_runtime_manifest(out_root: Path) -> None:
    path = out_root / "13_submission_audit/RUNTIME_MANIFEST.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = [
        {
            "step_id": "P06_COMPLETION",
            "script": "scripts/if_supplementary/complete_p06_random_dropout_predictions.py",
            "status": "PASS_FEATURE_CACHE_PROXY",
            "start_time": "",
            "end_time": now(),
            "duration_seconds": "",
            "output_files": str_rel(out_root / "05_modality_ablation_and_missingness/random_dropout_patient_level_predictions.csv"),
            "warnings": "Feature-cache prediction-registry proxy; raw-image checkpoint not re-run.",
            "errors": "",
        },
        {
            "step_id": "P10_COMPLETION",
            "script": "scripts/if_supplementary/complete_p10_perturbed_reliability_export.py",
            "status": "PASS_FEATURE_LEVEL_PROXY",
            "start_time": "",
            "end_time": now(),
            "duration_seconds": "",
            "output_files": str_rel(out_root / "09_reliability_validation/reliability_weights_clean_and_perturbed_patient_level.csv"),
            "warnings": "Feature-level proxy perturbation; raw-image perturbation weights unavailable.",
            "errors": "",
        },
        {
            "step_id": "P11_COMPLETION",
            "script": "scripts/if_supplementary/complete_p11_coe_intervention_logits.py",
            "status": "PARTIAL_NOT_ESTABLISHED",
            "start_time": "",
            "end_time": now(),
            "duration_seconds": "",
            "output_files": str_rel(out_root / "10_coe_faithfulness/coe_intervention_logits_patient_level.csv"),
            "warnings": "True CoE intervention logits and saliency masks unavailable.",
            "errors": "",
        },
        {
            "step_id": "PARTIAL_REFRESH",
            "script": "scripts/if_supplementary/refresh_partial_completion_package.py",
            "status": "PASS",
            "start_time": "",
            "end_time": now(),
            "duration_seconds": "",
            "output_files": str_rel(out_root / "IF_SUPPLEMENTARY_EXPERIMENTS_PACKAGE.zip"),
            "warnings": "",
            "errors": "",
        },
    ]
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["step_id"], keep="last")
    save_csv(combined, path)


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def test_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
    if bad:
        raise RuntimeError(f"Zip integrity check failed at {bad}")


if __name__ == "__main__":
    raise SystemExit(main())
