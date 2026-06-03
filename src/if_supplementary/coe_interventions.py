"""Conservative CoE intervention-logit completion for P11."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from .common import (
    PALETTE,
    diverging_cmap,
    display_center,
    input_paths,
    logit,
    now,
    read_csv,
    save_csv,
    save_figure,
    setup_style,
    str_rel,
    validate_no_raw_id_columns,
    write_text,
)
from .saliency_or_occlusion import visual_saliency_availability_summary


INTERVENTIONS = [
    {"condition": "clean_reference", "type": "reference", "step": "none", "control": "none", "repeat": 0},
    {"condition": "targeted_z1_clinical_prior_mask", "type": "targeted_step_mask", "step": "z1_clinical_prior", "control": "targeted", "repeat": 0},
    {"condition": "targeted_z2_colposcopy_mask", "type": "targeted_step_mask", "step": "z2_colposcopy", "control": "targeted", "repeat": 0},
    {"condition": "targeted_z3_oct_mask", "type": "targeted_step_mask", "step": "z3_oct", "control": "targeted", "repeat": 0},
    {"condition": "counterfactual_prior_swap", "type": "counterfactual_prior", "step": "z1_clinical_prior", "control": "targeted", "repeat": 0},
    {"condition": "visual_colposcopy_occlusion", "type": "visual_occlusion", "step": "z2_colposcopy", "control": "targeted", "repeat": 0},
    {"condition": "visual_oct_occlusion", "type": "visual_occlusion", "step": "z3_oct", "control": "targeted", "repeat": 0},
]


def run_p11_completion(
    out_root: Path,
    protocol: str = "strict_loco",
    clinical_interventions: bool = True,
    visual_interventions: bool = True,
    random_control_repeats: int = 5,
) -> dict[str, Path]:
    setup_style()
    out = out_root / "10_coe_faithfulness"
    out.mkdir(parents=True, exist_ok=True)
    paths = input_paths(out_root)
    coe = read_csv(paths["coe_proxy"], low_memory=False)
    patient = make_patient_level_intervention_table(
        coe,
        paths,
        protocol,
        clinical_interventions=clinical_interventions,
        visual_interventions=visual_interventions,
        random_control_repeats=random_control_repeats,
    )
    validate_no_raw_id_columns(patient)
    patient_path = save_csv(patient, out / "coe_intervention_logits_patient_level.csv")

    summary_path = save_csv(intervention_summary(patient), out / "coe_intervention_summary.csv")
    controls_path = save_csv(targeted_vs_random_controls(patient), out / "coe_targeted_vs_random_controls.csv")
    mono_path = save_csv(monotonicity_check(patient), out / "coe_monotonicity_check.csv")
    visual = visual_saliency_availability_summary(out)

    fig1 = plot_targeted_vs_random(patient, out / "figure_coe_targeted_vs_random_controls")
    fig2 = plot_monotonicity_status(patient, out / "figure_coe_counterfactual_monotonicity")
    fig3 = plot_step_specificity_heatmap(patient, out / "figure_coe_step_specificity_heatmap")
    fig4 = plot_visual_status(visual, out / "figure_coe_visual_saliency_vs_random")
    report_path = write_report(out, patient, visual)
    return {
        "patient_level": patient_path,
        "summary": summary_path,
        "targeted_vs_random": controls_path,
        "monotonicity": mono_path,
        "visual_summary": out / "coe_visual_intervention_summary.csv",
        "figure_targeted_data": fig1,
        "figure_monotonicity_data": fig2,
        "figure_heatmap_data": fig3,
        "figure_visual_data": fig4,
        "report": report_path,
    }


def make_patient_level_intervention_table(
    coe: pd.DataFrame,
    paths: dict[str, Path],
    protocol: str,
    clinical_interventions: bool,
    visual_interventions: bool,
    random_control_repeats: int,
) -> pd.DataFrame:
    base = coe.copy()
    if "patient_id_hash" not in base.columns:
        base["patient_id_hash"] = base["patient_id"].astype(str)
    specs = [INTERVENTIONS[0]]
    if clinical_interventions:
        specs.extend(INTERVENTIONS[1:5])
    if visual_interventions:
        specs.extend(INTERVENTIONS[5:])
    for rep in range(1, random_control_repeats + 1):
        specs.append({"condition": f"random_step_mask_control_{rep}", "type": "random_step_mask", "step": "random_step", "control": "random", "repeat": rep})

    created = now()
    rows = []
    for _, row in base.iterrows():
        clean_score = float(np.clip(pd.to_numeric(pd.Series([row["original_pred_score"]]), errors="coerce").fillna(0.5).iloc[0], 1e-6, 1 - 1e-6))
        for spec in specs:
            is_reference = spec["condition"] == "clean_reference"
            rows.append(
                {
                    "patient_id_hash": row["patient_id_hash"],
                    "patient_id_source_available": False,
                    "case_id_hash": row.get("case_id_hash", ""),
                    "center": row.get("center", ""),
                    "protocol": protocol,
                    "split": "held_out_test",
                    "condition": spec["condition"],
                    "intervention_type": spec["type"],
                    "intervention_step": spec["step"],
                    "control_type": spec["control"],
                    "random_control_repeat": int(spec["repeat"]),
                    "y_cin2": int(row.get("y_cin2", 0)),
                    "y_cin3": int(row.get("y_cin3", 0)),
                    "clean_pred_score": clean_score,
                    "clean_final_logit": float(logit(clean_score)),
                    "clean_z1_logit": row.get("original_z1_logit", np.nan),
                    "clean_z2_logit": row.get("original_z2_logit", np.nan),
                    "clean_z3_logit": row.get("original_z3_logit", np.nan),
                    "intervened_pred_score": clean_score if is_reference else np.nan,
                    "intervened_final_logit": float(logit(clean_score)) if is_reference else np.nan,
                    "intervened_z1_logit": row.get("original_z1_logit", np.nan) if is_reference else np.nan,
                    "intervened_z2_logit": row.get("original_z2_logit", np.nan) if is_reference else np.nan,
                    "intervened_z3_logit": row.get("original_z3_logit", np.nan) if is_reference else np.nan,
                    "delta_pred_score": 0.0 if is_reference else np.nan,
                    "delta_final_logit": 0.0 if is_reference else np.nan,
                    "delta_z1_logit": 0.0 if is_reference else np.nan,
                    "delta_z2_logit": 0.0 if is_reference else np.nan,
                    "delta_z3_logit": 0.0 if is_reference else np.nan,
                    "saliency_available": False,
                    "saliency_mask_path": "",
                    "raw_image_intervention_executed": False,
                    "feature_level_proxy_intervention": False,
                    "intervention_logit_available": bool(is_reference),
                    "status": "REFERENCE_ONLY" if is_reference else "NOT_EXECUTABLE",
                    "source_coe_file": str_rel(paths["coe_proxy"]),
                    "source_checkpoint": "not_loaded_no_saved_intervention_logits",
                    "script": "scripts/if_supplementary/complete_p11_coe_intervention_logits.py",
                    "created_at": created,
                    "notes": "No saved CoE intervention inference logits or saliency masks are available; faithfulness is not established.",
                }
            )
    return pd.DataFrame(rows)


def intervention_summary(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, intervention_type, control_type), g in patient.groupby(["condition", "intervention_type", "control_type"], dropna=False):
        rows.append(
            {
                "condition": condition,
                "intervention_type": intervention_type,
                "control_type": control_type,
                "n": int(len(g)),
                "intervention_logit_available_n": int(g["intervention_logit_available"].sum()),
                "raw_image_intervention_executed": bool(g["raw_image_intervention_executed"].any()),
                "saliency_available": bool(g["saliency_available"].any()),
                "mean_delta_pred_score": float(pd.to_numeric(g["delta_pred_score"], errors="coerce").mean()),
                "status": "REFERENCE_ONLY" if condition == "clean_reference" else "NOT_ESTABLISHED",
                "claim_boundary": "No CoE faithfulness or causal explanation claim is supported.",
            }
        )
    return pd.DataFrame(rows)


def targeted_vs_random_controls(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control_type, g in patient.groupby("control_type", dropna=False):
        sub = g[~g["condition"].eq("clean_reference")]
        rows.append(
            {
                "control_type": control_type,
                "n_conditions": int(sub["condition"].nunique()),
                "n_patient_condition_rows": int(len(sub)),
                "valid_delta_rows": int(pd.to_numeric(sub["delta_pred_score"], errors="coerce").notna().sum()),
                "mean_abs_delta_pred_score": float(pd.to_numeric(sub["delta_pred_score"], errors="coerce").abs().mean()),
                "status": "NOT_ESTABLISHED" if control_type != "none" else "REFERENCE_ONLY",
                "notes": "Targeted-vs-random contrast is not executable without intervened logits.",
            }
        )
    return pd.DataFrame(rows)


def monotonicity_check(patient: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "targeted_step_counterfactual_monotonicity",
                "n_patient_condition_rows": int(len(patient[~patient["condition"].eq("clean_reference")])),
                "valid_delta_rows": int(pd.to_numeric(patient["delta_pred_score"], errors="coerce").notna().sum() - patient["condition"].eq("clean_reference").sum()),
                "monotonicity_established": False,
                "status": "NOT_EXECUTABLE",
                "blocker": "Intervened CoE logits are unavailable.",
            }
        ]
    )


def plot_targeted_vs_random(patient: pd.DataFrame, stem: Path) -> Path:
    data = intervention_summary(patient)
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    counts = data.groupby("status")["condition"].count().reset_index(name="n_conditions")
    ax.bar(counts["status"], counts["n_conditions"], color=[PALETTE[1], PALETTE[4]][: len(counts)])
    ax.set_title("CoE Intervention Availability")
    ax.set_xlabel("")
    ax.set_ylabel("Conditions")
    for i, row in counts.iterrows():
        ax.text(i, row["n_conditions"] + 0.08, str(int(row["n_conditions"])), ha="center", va="bottom", color="#30335f")
    save_figure(fig, stem)
    return data_path


def plot_monotonicity_status(patient: pd.DataFrame, stem: Path) -> Path:
    data = monotonicity_check(patient)
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(["Valid intervention deltas", "Missing intervention deltas"], [0, int(data["n_patient_condition_rows"].iloc[0])], color=[PALETTE[0], PALETTE[4]])
    ax.set_title("Counterfactual Monotonicity Check")
    ax.set_xlabel("Patient-condition rows")
    ax.set_ylabel("")
    save_figure(fig, stem)
    return data_path


def plot_step_specificity_heatmap(patient: pd.DataFrame, stem: Path) -> Path:
    clean = patient[patient["condition"].eq("clean_reference")].copy()
    clean["center_display"] = clean["center"].map(display_center)
    heat = clean.groupby("center_display")[["clean_z1_logit", "clean_z2_logit", "clean_z3_logit"]].mean()
    data = heat.reset_index()
    data_path = save_csv(data, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    if sns is not None:
        sns.heatmap(heat, cmap=diverging_cmap(), center=0, linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "Clean proxy logit"})
    else:
        ax.imshow(heat.fillna(0), aspect="auto")
        ax.set_xticks(range(len(heat.columns)), heat.columns)
        ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Clean CoE Proxy State Map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, stem)
    return data_path


def plot_visual_status(visual: pd.DataFrame, stem: Path) -> Path:
    data_path = save_csv(visual, stem.with_name(stem.name + "_data.csv"))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.barh(visual["visual_intervention_type"], [0 for _ in range(len(visual))], color=PALETTE[0])
    ax.scatter([0 for _ in range(len(visual))], visual["visual_intervention_type"], s=120, color=PALETTE[4], edgecolor="#333333")
    ax.set_xlim(-0.05, 1.0)
    ax.set_title("Visual Saliency and Occlusion Availability")
    ax.set_xlabel("Executable intervention evidence")
    ax.set_ylabel("")
    ax.set_xticks([0, 1], ["Unavailable", "Available"])
    save_figure(fig, stem)
    return data_path


def write_report(out: Path, patient: pd.DataFrame, visual: pd.DataFrame) -> Path:
    lines = [
        "# CoE Faithfulness Report",
        "",
        "## P11 Completion Status",
        "",
        "Status: `PARTIAL_NOT_ESTABLISHED`.",
        "",
        "A patient-level CoE intervention-logit table was exported, but only the clean reference logits are available.",
        "Targeted, random-control, counterfactual, and visual saliency/occlusion intervention logits are not present in the locked outputs.",
        "Therefore CoE faithfulness and causal explanation claims remain unsupported.",
        "",
        f"Patient-condition rows: `{len(patient)}`.",
        f"Unique patients: `{patient['patient_id_hash'].nunique()}`.",
        "",
        "## Visual Intervention Availability",
        "",
        visual.to_string(index=False),
        "",
        "## Claim Boundary",
        "",
        "- Allowed: clean CoE proxy state visualization and explicit audit of missing intervention evidence.",
        "- Not allowed: CoE faithfulness, causal explanation, or saliency-grounded claims.",
    ]
    return write_text(out / "COE_FAITHFULNESS_REPORT.md", "\n".join(lines) + "\n")
