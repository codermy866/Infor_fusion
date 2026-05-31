#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

OUT = C.OUT / "coe01_decoder_verification"
LOCO = C.OUT / "loco01_hydra_vlm_loco"


def label_case(row) -> str:
    pred = bool(row["pred_positive"])
    y = int(row["cin3_label"])
    if pred and y:
        return "true_positive"
    if pred and not y:
        return "false_positive"
    if (not pred) and y:
        return "false_negative"
    return "true_negative"


def make_coe(row, meta) -> str:
    risk = "high" if row["score_cin2plus"] >= row["selected_threshold"] else "low"
    hpv = str(meta.get("hpv16_18_status", "unavailable"))
    tct = str(meta.get("tct_status_harmonized", "unavailable"))
    age = meta.get("age", "unavailable")
    return (
        f"Template CoE: the cached multimodal score is {row['score_cin2plus']:.3f}, "
        f"which is {'above' if risk == 'high' else 'below'} the locked fold threshold. "
        f"Structured priors include age={age}, HPV16/18={hpv}, TCT={tct}. "
        "This text is a transparency aid generated from available fields, not a clinically validated explanation."
    )


def main() -> None:
    C.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (LOCO / "patient_level_predictions.csv").exists():
        C.write_text(OUT / "coe01_audit_report.md", "# COE01 Audit Report\n\nStatus: `BLOCKED`. LOCO01 predictions unavailable.\n")
        C.status_json(OUT / "status.json", "BLOCKED", "LOCO01 predictions unavailable.")
        return
    pred = pd.read_csv(LOCO / "patient_level_predictions.csv")
    dl = C.load_data_lock()
    meta = dl.set_index("case_id")
    pred["case_type"] = pred.apply(label_case, axis=1)
    rows = []
    for ctype, g in pred.groupby("case_type"):
        take = g.sort_values("score_cin2plus", ascending=ctype in {"false_negative", "true_negative"}).head(5)
        for _, r in take.iterrows():
            m = meta.loc[r["case_id"]] if r["case_id"] in meta.index else {}
            rows.append(
                {
                    "case_type": ctype,
                    "patient_id": r["patient_id"],
                    "case_id": r["case_id"],
                    "centre": r["centre"],
                    "centre_label": r["centre_label"],
                    "score_cin2plus": r["score_cin2plus"],
                    "threshold": r["selected_threshold"],
                    "cin2_label": r["cin2_label"],
                    "cin3_label": r["cin3_label"],
                    "coe_text": make_coe(r, m),
                    "expert_validation": "NOT_AVAILABLE",
                }
            )
    examples = pd.DataFrame(rows)
    hard = examples[examples["centre_label"].eq("Xiangyang")].copy()
    consistency = pd.DataFrame(
        [
            {
                "metric": "risk_direction_matches_prediction",
                "value": 1.0,
                "notes": "Template CoE risk direction is mechanically tied to locked model prediction.",
            },
            {
                "metric": "expert_validation_available",
                "value": 0.0,
                "notes": "No expert faithfulness ratings available.",
            },
            {
                "metric": "clinical_faithfulness_claim_allowed",
                "value": 0.0,
                "notes": "Forbidden without independent expert validation.",
            },
        ]
    )
    C.write_csv(OUT / "coe_case_examples.csv", examples)
    C.write_csv(OUT / "coe_proxy_consistency_metrics.csv", consistency)
    C.write_csv(OUT / "coe_hard_centre_examples.csv", hard)
    C.write_text(
        OUT / "coe_limitation_statement.txt",
        "CoE outputs in this recovery package are template-supervised transparency aids. Expert validation is not available; therefore clinical faithfulness must not be claimed.\n",
    )
    report = [
        "# COE01 Decoder Verification Report",
        "",
        "Status: `COMPLETED_WITH_LIMITATION`",
        "",
        "EXPERT_VALIDATION = `NOT_AVAILABLE`",
        "",
        "No VLM decoder faithfulness claim is supported. CoE may be described only as a transparency aid.",
        "",
        "## Proxy Metrics",
        "",
        C.md_table(consistency),
        "",
        "## Example Counts",
        "",
        C.md_table(examples.groupby("case_type").size().reset_index(name="n")),
    ]
    C.write_text(OUT / "coe01_audit_report.md", "\n".join(report) + "\n")
    C.status_json(OUT / "status.json", "PASS", "CoE examples collected; expert validation unavailable.")
    C.file_manifest(OUT, OUT / "coe01_file_manifest.csv")


if __name__ == "__main__":
    main()
