# Repository and Data Audit

Created at: `2026-06-02 16:39:49`

## Analytic Cohort
- Candidate analytic cohort size: `1897` patients/cases from `outputs/publishable_v2/data_lock/data_lock_n1897.csv`.
- Available centers: Shiyan, Enshi, Wuhan Renmin, Jingzhou, Xiangyang.
- Patient ID column: `patient_id`; case ID column: `case_id`.
- Endpoint labels: `pathology_cin2plus`, `pathology_cin3plus`.
- Invasive cancer as a separate patient-level label: `not available`; CIN3+ is evaluated from `pathology_cin3plus`.

## Modality Availability
| center | center_display | n | cin2_pos | cin3_pos | oct_available_n | colposcopy_available_n | clinical_prior_available_n | cin2_prevalence | cin3_prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 十堰市人民医院 | Shiyan | 496 | 28 | 4 | 496 | 496 | 496 | 0.05645 | 0.008065 |
| 恩施州中心医院 | Enshi | 406 | 130 | 64 | 406 | 406 | 406 | 0.3202 | 0.1576 |
| 武大人民医院 | Wuhan Renmin | 89 | 89 | 49 | 89 | 89 | 89 | 1 | 0.5506 |
| 荆州市第一人民医院 | Jingzhou | 406 | 37 | 13 | 406 | 406 | 406 | 0.09113 | 0.03202 |
| 襄阳市中心医院 | Xiangyang | 500 | 110 | 61 | 500 | 500 | 500 | 0.22 | 0.122 |

## Protocol Status
- Primary strict five-fold LOCO can be reconstructed from current patient-level prediction files.
- Fixed external split files exist and are preserved as secondary supplementary material only.
- Validation predictions contain source-center validation rows; this package locks one hard validation center per LOCO fold for thresholds.

## Blockers
- True invasive-cancer-only endpoint cannot be computed without a separate invasive cancer label.
- Raw intervention-based CoE faithfulness cannot be computed without saved intervention logits or an inference hook exporting intervened states.
- Image quality annotations are not available; reliability quality analyses use proxy variables only.
- Random 10/30/50% modality dropout patient-level predictions are not available in the locked output tree.
