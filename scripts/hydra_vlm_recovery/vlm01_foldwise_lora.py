#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

spec = importlib.util.spec_from_file_location("hvr_common", Path(__file__).with_name("00_common.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


OUT = C.OUT / "vlm01_foldwise_lora"
P00 = C.OUT / "p00_protocol_lock"


def _pca_fit_transform(x_src: np.ndarray, x_tgt: np.ndarray, n: int, seed: int):
    scaler = StandardScaler()
    xs = scaler.fit_transform(np.nan_to_num(x_src))
    xt = scaler.transform(np.nan_to_num(x_tgt))
    k = max(1, min(n, xs.shape[0] - 1, xs.shape[1]))
    pca = PCA(n_components=k, random_state=seed)
    ps = pca.fit_transform(xs)
    pt = pca.transform(xt)
    return ps.astype(np.float32), pt.astype(np.float32), float(np.sum(pca.explained_variance_ratio_))


def _frozen_slice(x_src: np.ndarray, x_tgt: np.ndarray, n: int):
    scaler = StandardScaler()
    xs = scaler.fit_transform(np.nan_to_num(x_src))
    xt = scaler.transform(np.nan_to_num(x_tgt))
    k = min(n, xs.shape[1])
    return xs[:, :k].astype(np.float32), xt[:, :k].astype(np.float32)


def _text_features(src_meta: pd.DataFrame, tgt_meta: pd.DataFrame):
    cols = ["hpv16_18_status", "tct_status_harmonized", "hpv_status_harmonized"]
    for c in cols:
        if c not in src_meta.columns:
            src_meta[c] = "missing"
            tgt_meta[c] = "missing"
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=None)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    xs_cat = enc.fit_transform(src_meta[cols].fillna("missing").astype(str))
    xt_cat = enc.transform(tgt_meta[cols].fillna("missing").astype(str))
    age_scaler = StandardScaler()
    xs_age = age_scaler.fit_transform(src_meta[["age"]].fillna(src_meta["age"].median()))
    xt_age = age_scaler.transform(tgt_meta[["age"]].fillna(src_meta["age"].median()))
    xs = np.hstack([xs_age, xs_cat]).astype(np.float32)
    xt = np.hstack([xt_age, xt_cat]).astype(np.float32)
    return xs, xt


def _add_feature_cols(base: pd.DataFrame, prefix: str, arr: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    for j in range(arr.shape[1]):
        out[f"{prefix}_{j:03d}"] = arr[:, j]
    return out


def _base_meta(meta: pd.DataFrame, fold_id: str, role: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": meta["case_id"].astype(str),
            "original_patient_id": meta["patient_id"].astype(str),
            "case_id": meta["case_id"].astype(str),
            "centre": meta["center_name"].astype(str),
            "centre_label": meta["centre_label"].astype(str),
            "fold_id": fold_id,
            "split_role": role,
            "cin2_label": meta["pathology_cin2plus"].astype(int),
            "cin3_label": meta["pathology_cin3plus"].astype(int),
            "feature_source_type": "CACHED_FEATURE_ADAPTER",
            "lora_status": "NOT_RUN",
        }
    )


def main() -> None:
    C.ensure_dirs()
    status = C.read_json(P00 / "status.json")
    if status["status"] not in {"PASS", "FAILED_PARTIAL"}:
        raise SystemExit("P00 did not pass; VLM01 blocked.")

    folds = C.read_json(P00 / "locked_folds.json")
    dl, arrays = C.load_feature_arrays()
    overall = []
    dim_rows = []
    parquet_backends = []
    for fidx, f in enumerate(folds, start=1):
        fold_dir = OUT / f"fold_{f['held_out_centre_label']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        src_mask = dl["center_name"].isin(f["source_centres"]).to_numpy()
        tgt_mask = dl["center_name"].eq(f["target_centre"]).to_numpy()
        src_meta = dl.loc[src_mask].reset_index(drop=True)
        tgt_meta = dl.loc[tgt_mask].reset_index(drop=True)

        seed = 2026 + fidx
        col_s, col_t, col_var = _pca_fit_transform(arrays["col"][src_mask], arrays["col"][tgt_mask], 48, seed)
        oct_s, oct_t, oct_var = _pca_fit_transform(arrays["oct"][src_mask], arrays["oct"][tgt_mask], 48, seed)
        cli_s, cli_t, cli_var = _pca_fit_transform(arrays["clinical"][src_mask], arrays["clinical"][tgt_mask], 24, seed)
        txt_s, txt_t = _text_features(src_meta.copy(), tgt_meta.copy())
        fcol_s, fcol_t = _frozen_slice(arrays["col"][src_mask], arrays["col"][tgt_mask], 48)
        foct_s, foct_t = _frozen_slice(arrays["oct"][src_mask], arrays["oct"][tgt_mask], 48)
        fcli_s, fcli_t = _frozen_slice(arrays["clinical"][src_mask], arrays["clinical"][tgt_mask], 24)

        src = _base_meta(src_meta, f["fold_id"], "source")
        tgt = _base_meta(tgt_meta, f["fold_id"], "target")
        for df_name, df_obj, mats in [
            ("source", src, (col_s, oct_s, cli_s, txt_s, fcol_s, foct_s, fcli_s)),
            ("target", tgt, (col_t, oct_t, cli_t, txt_t, fcol_t, foct_t, fcli_t)),
        ]:
            a, b, c, d, e, g, h = mats
            df_obj = _add_feature_cols(df_obj, "colpo_feature", a)
            df_obj = _add_feature_cols(df_obj, "oct_feature", b)
            df_obj = _add_feature_cols(df_obj, "clinical_feature", c)
            df_obj = _add_feature_cols(df_obj, "text_feature", d)
            df_obj = _add_feature_cols(df_obj, "frozen_colpo_feature", e)
            df_obj = _add_feature_cols(df_obj, "frozen_oct_feature", g)
            df_obj = _add_feature_cols(df_obj, "frozen_clinical_feature", h)
            backend = C.save_table(df_obj, fold_dir / f"{df_name}_features.parquet")
            parquet_backends.append(backend)

        manifest = pd.DataFrame(
            [
                {"modality": "colposcopy", "feature_prefix": "colpo_feature", "n_features": col_s.shape[1], "source_fit_only": True, "explained_variance": col_var},
                {"modality": "oct", "feature_prefix": "oct_feature", "n_features": oct_s.shape[1], "source_fit_only": True, "explained_variance": oct_var},
                {"modality": "clinical", "feature_prefix": "clinical_feature", "n_features": cli_s.shape[1], "source_fit_only": True, "explained_variance": cli_var},
                {"modality": "text_semantic", "feature_prefix": "text_feature", "n_features": txt_s.shape[1], "source_fit_only": True, "explained_variance": "NA"},
                {"modality": "frozen_colposcopy", "feature_prefix": "frozen_colpo_feature", "n_features": fcol_s.shape[1], "source_fit_only": False, "explained_variance": "NA"},
                {"modality": "frozen_oct", "feature_prefix": "frozen_oct_feature", "n_features": foct_s.shape[1], "source_fit_only": False, "explained_variance": "NA"},
                {"modality": "frozen_clinical", "feature_prefix": "frozen_clinical_feature", "n_features": fcli_s.shape[1], "source_fit_only": False, "explained_variance": "NA"},
            ]
        )
        C.write_csv(fold_dir / "feature_manifest.csv", manifest)
        dim_rows.extend([{"fold_id": f["fold_id"], **r} for r in manifest.to_dict("records")])

        config = {
            "fold_id": f["fold_id"],
            "held_out_centre": f["held_out_centre"],
            "feature_source_type": "CACHED_FEATURE_ADAPTER",
            "biomedclip_lora_status": "NOT_RUN",
            "reason_lora_not_run": "No auditable BioMedCLIP-LoRA training stack/checkpoint was available in this recovery pass; source-only PCA/standardisation adapters were fitted to cached locked feature arrays.",
            "parquet_backend": sorted(set(parquet_backends)),
            "target_labels_used_for_adapter_training": False,
        }
        C.write_json(fold_dir / "lora_or_adapter_config.json", config)
        C.write_csv(
            fold_dir / "training_log.csv",
            pd.DataFrame(
                [
                    {
                        "stage": "source_only_cached_feature_adapter",
                        "status": "COMPLETED",
                        "n_source": len(src_meta),
                        "n_target": len(tgt_meta),
                        "target_labels_used": False,
                    }
                ]
            ),
        )
        leakage = f"""# VLM01 Leakage Audit: {f['held_out_centre_label']}

Status: `PASS_WITH_CACHED_FEATURE_ADAPTER`

- Source centres: {', '.join(f['source_centres'])}
- Target centre: {f['target_centre']}
- Adapter fitting used source-centre cached features only.
- Target labels were written only for downstream evaluation and were not used for adapter fitting.
- BioMedCLIP-LoRA was not run and is not claimed.
- `patient_id` in feature files is the unique locked `case_id`; `original_patient_id` preserves the source patient identifier.
"""
        C.write_text(fold_dir / "leakage_audit.md", leakage)
        C.status_json(
            fold_dir / "status.json",
            "FAILED_PARTIAL",
            "BioMedCLIP-LoRA not run; source-only cached feature adapter generated.",
            feature_source_type="CACHED_FEATURE_ADAPTER",
        )
        overall.append(
            {
                "fold_id": f["fold_id"],
                "held_out_centre": f["held_out_centre"],
                "status": "FAILED_PARTIAL",
                "feature_source_type": "CACHED_FEATURE_ADAPTER",
                "n_source": len(src_meta),
                "n_target": len(tgt_meta),
                "biomedclip_lora_verified": False,
            }
        )

    C.write_csv(OUT / "vlm01_overall_status.csv", pd.DataFrame(overall))
    C.write_csv(OUT / "vlm01_feature_dimension_summary.csv", pd.DataFrame(dim_rows))
    C.write_text(
        OUT / "vlm01_leakage_control_report.md",
        "# VLM01 Leakage Control Report\n\nAll fold-specific adapters were fitted on source-centre cached features only. Target labels were stored only for downstream evaluation. BioMedCLIP-LoRA was not run and is not verified.\n",
    )
    C.write_text(
        OUT / "vlm01_audit_report.md",
        "# VLM01 Audit Report\n\nStatus: `FAILED_PARTIAL`.\n\nAt least one feature file was generated per fold, but the feature source is `CACHED_FEATURE_ADAPTER`, not `BIOMEDCLIP_LORA`. Parquet writing used pandas fallback if no parquet engine was available.\n",
    )
    C.file_manifest(OUT, OUT / "vlm01_file_manifest.csv")


if __name__ == "__main__":
    main()
