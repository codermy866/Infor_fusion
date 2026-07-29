# Shift-aware Latent Active Fusion

This repository contains the current Information Fusion 2026 method code for
patient-level sequential evidence acquisition across clinical variables,
colposcopy and OCT in a five-centre cervical-lesion cohort.

## Current method

The current v3 implementation combines:

- shared/private multimodal latent representations;
- source-centre adversarial invariance and a private centre audit head;
- precision-weighted latent fusion;
- heteroscedastic counterfactual value-of-information estimates;
- a lower-confidence-bound acquisition policy;
- multitask CIN2+ and CIN3+ prediction;
- a source-calibrated joint AUPRC-retention and CIN3+ sensitivity constraint;
- an optional frozen VLM paired-concept expert that becomes available only
  after colposcopy acquisition;
- BiCER evidence-response regularisation.

The implementation is a retrospective acquisition simulation. Relative costs
are sensitivity weights, not measured monetary, workflow or patient-burden
costs. Source-calibrated CIN3+ constraints are not guarantees under target
centre shift.

## Repository scope

Only the latest v3 code and its minimal compatibility dependencies are kept in
the published snapshot. Data, patient metadata, predictions, figures, model
weights, checkpoints, development outputs and rejected Qwen3-VL scripts are
excluded.

`models/active_fusion_v2.py` and `scripts/active_fusion_v2/run_v2.py` are
retained only as internal base/runtime dependencies of v3; the published
entrypoint is `scripts/active_fusion_v3/run_v3.py`.

## Environment

The formal local environment used by the project is:

```text
/data2/hmy_pri/VLM_Caus_Rm_Mics/my_retfound
```

Install portable dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Frozen VLM feature extraction

Download the official BiomedCLIP files into a local model directory and run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/active_fusion_v3/extract_biomedclip_concepts.py \
  --model-dir /path/to/biomedclip_model \
  --input /path/to/data_lock.csv \
  --output /path/to/paired_concepts.pt
```

The extractor projects input rows to identifiers and colposcopy paths before
inference. Centre and pathology outcome fields are not passed to the VLM.

## Source-only development

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/active_fusion_v3/run_v3.py \
  --mode development \
  --development-all-folds \
  --arm v3_latent_shared_only \
  --concept-cache /path/to/paired_concepts.pt \
  --output /path/to/development_output
```

## Locked formal evaluation

Formal mode requires all five locked folds and at least three seeds:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/active_fusion_v3/run_v3.py \
  --mode formal \
  --arm v3_latent_shared_only \
  --formal-seeds 2026 2027 2028 \
  --concept-cache /path/to/paired_concepts.pt \
  --output /path/to/formal_output
```

Target outcomes are loaded only after source training, checkpoint selection,
acquisition-cost selection and safety-threshold calibration are frozen.

## Verification

```bash
python -m unittest tests.test_active_fusion_v3 -v
python -m py_compile \
  models/active_fusion_v2.py \
  models/active_fusion_v3.py \
  scripts/active_fusion_v2/run_v2.py \
  scripts/active_fusion_v3/*.py
```
