# Experiment Scope

Current HyDRA-CoE experiments use only colposcopy images, OCT images, and HPV/TCT/Age clinical text variables. Examination reports, diagnosis reports, generated reports, and VLM evidence caches are not used in the main experiment.

The locked main cohort is the final 1,897-case tri-modal alignment derived from the original 3,010-case registry. The approximately 130k image/B-scan volume is represented by colposcopy image paths and OCT image/B-scan paths in `paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv`.

The current experimental boundary excludes examination-report text, report-derived supervision, generated-report targets, and VLM evidence-cache retrieval. The following fields and resources are not used as model inputs, training targets, evaluation inputs, or generated outputs in current experiments:

- `report`
- `report_text`
- `clinical_report`
- `diagnosis_report`
- `generated_report`
- `exam_report`
- `examination_report`
- `vlm_json_path`
- `use_vlm_retriever`
- VLM evidence caches
- `检查报告`
- `诊断报告`

Allowed inputs are:

- OCT image features or OCT image tensors
- Colposcopy image features or colposcopy image tensors
- `clinical_features` built only from normalized age, HPV, and TCT
- `clinical_info` strings with the exact form `HPV: <normalized_hpv>, TCT: <normalized_tct>, Age: <age>`
- `center_idx` or equivalent center metadata
- `label`, used only for supervised training and evaluation

Archived-report availability may be described in cohort tables as a source-data characteristic, but those report files are not read by the training or evaluation pipeline. Pathology labels are not inserted into `clinical_info`.
