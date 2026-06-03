"""Saliency and visual-occlusion availability audit for P11."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import save_csv


def visual_saliency_availability_summary(out: Path) -> pd.DataFrame:
    """Return a conservative availability table for visual CoE interventions."""
    rows = [
        {
            "visual_intervention_type": "colposcopy_occlusion",
            "saliency_available": False,
            "occlusion_prediction_available": False,
            "raw_image_intervention_executed": False,
            "status": "NOT_EXECUTABLE",
            "blocker": "No saved saliency masks or occlusion-inference logits were available in the locked outputs.",
        },
        {
            "visual_intervention_type": "oct_occlusion",
            "saliency_available": False,
            "occlusion_prediction_available": False,
            "raw_image_intervention_executed": False,
            "status": "NOT_EXECUTABLE",
            "blocker": "No saved saliency masks or occlusion-inference logits were available in the locked outputs.",
        },
        {
            "visual_intervention_type": "random_visual_mask_control",
            "saliency_available": False,
            "occlusion_prediction_available": False,
            "raw_image_intervention_executed": False,
            "status": "NOT_EXECUTABLE",
            "blocker": "Random visual controls require the same unavailable occlusion-inference pipeline.",
        },
    ]
    df = pd.DataFrame(rows)
    save_csv(df, out / "coe_visual_intervention_summary.csv")
    return df

