#!/usr/bin/env python3
"""Quality and source-only predictive audit for structured VLM concepts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT))

from paper_revision.scripts.clinical_variable_mapping import clinical_features_from_row  # noqa: E402
from scripts.active_fusion_v2.run_v2 import (  # noqa: E402
    DEFAULT_SPLIT_ROOT,
    derive_source_train_validation,
    discover_folds,
    load_frame,
    write_json,
)


def key(row: pd.Series) -> str:
    return f"{row['patient_id']}||{row['oct_id']}"


def fit_score(train_x, train_y, val_x, val_y):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=20260729,
        ),
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(val_x)[:, 1]
    return {
        "auroc": float(roc_auc_score(val_y, probability)),
        "auprc": float(average_precision_score(val_y, probability)),
        "brier": float(brier_score_loss(val_y, probability)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/shared/"
            "qwen3vl_structured_colposcopy_concepts.pt"
        ),
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Audit every locked outer fold using source-only inner splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/shift_safe_vlm_v3_20260729/analysis/"
            "structured_concept_source_audit.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    features = payload["features"]
    if len(features) != 1897 or payload.get("failures"):
        raise ValueError("concept cache is not complete and failure-free")
    medical = np.stack([value["medical"].numpy() for value in features.values()])
    generic = np.stack([value["generic"].numpy() for value in features.values()])
    sorted_keys = sorted(
        features,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    shuffled = {
        value: sorted_keys[(index + 1) % len(sorted_keys)]
        for index, value in enumerate(sorted_keys)
    }
    quality = {}
    for name, matrix in (("medical", medical), ("generic", generic)):
        quality[name] = {
            "n": len(matrix),
            "dimensions": matrix.shape[1],
            "unique_rows": int(len(np.unique(matrix, axis=0))),
            "mean": matrix.mean(axis=0).tolist(),
            "sd": matrix.std(axis=0).tolist(),
            "unique_values_per_dimension": [
                int(len(np.unique(matrix[:, index])))
                for index in range(matrix.shape[1])
            ],
        }

    def matrix(frame: pd.DataFrame, mode: str):
        clinical = np.stack(
            [clinical_features_from_row(row) for _, row in frame.iterrows()]
        )
        if mode == "clinical":
            return clinical
        rows = []
        for _, row in frame.iterrows():
            case = key(row)
            if mode == "medical":
                concept = features[case]["medical"].numpy()
            elif mode == "generic":
                concept = features[case]["generic"].numpy()
            elif mode == "shuffled":
                concept = features[shuffled[case]]["medical"].numpy()
            else:
                raise ValueError(mode)
            rows.append(np.r_[clinical[len(rows)], concept])
        return np.stack(rows)

    folds = discover_folds(args.split_root)
    if not args.all_folds:
        folds = folds[:1]
    fold_probes = {}
    split_audits = {}
    for fold in folds:
        train, validation, split_audit = derive_source_train_validation(
            load_frame(fold / "train_labels.csv"),
            load_frame(fold / "val_labels.csv"),
            seed=args.seed,
        )
        train_y = train.pathology_cin2plus.to_numpy(dtype=int)
        val_y = validation.pathology_cin2plus.to_numpy(dtype=int)
        fold_probes[fold.name] = {
            mode: fit_score(
                matrix(train, mode),
                train_y,
                matrix(validation, mode),
                val_y,
            )
            for mode in ("clinical", "medical", "generic", "shuffled")
        }
        split_audits[fold.name] = split_audit
    aggregate = {}
    for mode in ("clinical", "medical", "generic", "shuffled"):
        aggregate[mode] = {}
        for metric in ("auroc", "auprc", "brier"):
            values = np.asarray(
                [
                    fold_probes[fold.name][mode][metric]
                    for fold in folds
                ],
                dtype=float,
            )
            aggregate[mode][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    result = {
        "schema": "structured_vlm_concept_source_audit_v1",
        "cache_schema": payload.get("schema"),
        "cache_n": len(features),
        "failure_n": len(payload.get("failures", {})),
        "outcome_fields_passed_to_model": payload.get(
            "outcome_fields_passed_to_model"
        ),
        "quality": quality,
        "source_folds": [fold.name for fold in folds],
        "source_split_audits": split_audits,
        "probe_definition": (
            "class-weighted logistic regression; clinical versus clinical plus "
            "seven concepts; source validation only"
        ),
        "fold_probes": fold_probes,
        "aggregate": aggregate,
        "claim_boundary": (
            "This is a source-only linear probe for screening a concept route, "
            "not a formal external-centre result."
        ),
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
