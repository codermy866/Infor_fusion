# HyDRA-CoE 403例校正外部测试 Clean Rerun 与终稿实验收口 Prompts

## 0. 当前状态判定

根据当前执行汇总，A–Q 的代码实现基本完成，但实验产物尚未形成“终稿级干净闭环”。主要问题不是缺少脚本，而是结果口径不统一。

当前主要问题：

```text
1. final_pipeline_manifest.csv 全部为 dry_run，没有真实训练链路记录。
2. J–O 多数结果表混有旧外部测试口径，n=148 / 196 / 283 / 403 混杂。
3. 最新可信主结果来自 real_50epoch_5center_corrected，但后续消融、鲁棒性、LOCO、CoE、临床决策尚未全部用该口径重跑。
4. Stage-1 clinical semantic adapter 已有 checkpoint，但是否真实接入 Stage-2 主模型训练仍需程序级确认。
5. 403例校正外部测试应成为终稿统一 external_test 口径。
```

因此，下一阶段不应继续扩展新模块，而应执行：

```text
Clean rerun under corrected 5-center 1897-patient cohort and 403-case external test.
```

---

## 1. 终稿统一实验口径

最终所有主文实验结果必须同时满足：

```text
Final aligned cohort: 1897 patients
Train/Val/Test: 1317 / 177 / 403
External test N: 403
Input modalities: colposcopy + OCT + HPV/TCT/Age
No examination reports
No VLM evidence cache
No test-set threshold tuning
All main tables use internal-validation locked threshold
All manifest entries are real runs, not dry_run
All method names are HyDRA-CoE-facing, not Bio-COT-facing
```

任何 `n=148`、`n=196`、`n=283` 的结果只能进入历史归档或补充审计，不应进入终稿主表。

---

## 2. Clean rerun 总执行顺序

请按以下顺序执行：

```text
R0. Archive legacy mixed-n results and freeze current status
R1. Verify corrected cohort, split, and feature cache before rerun
R2. Confirm Stage-1 clinical semantic adapter is loaded in Stage-2
R3. Real rerun of Full HyDRA-CoE on corrected 403 external test
R4. Rerun baselines under the same corrected 403 protocol
R5. Rerun requirement-level ablations under corrected 403 protocol
R6. Rerun missing-modality and corruption robustness using corrected full checkpoint
R7. Rebuild label-noise stress test under corrected 403 protocol
R8. Rebuild LOCO and center-wise calibration with clean manifest
R9. Rebuild CoE faithfulness proxy using corrected full checkpoints
R10. Rebuild clinical decision quality from corrected 403 predictions
R11. Build final paper-ready tables from corrected-only results
R12. Replace dry_run final manifest with real clean-rerun manifest
R13. Final paper-facing naming and method separation
```

---

# Prompt R0. Archive legacy results and freeze current status

```text
请在 codermy866/Infor_fusion 仓库中执行一次旧结果归档与当前状态冻结。目标是避免 n=148/196/283 的旧表继续污染 403 例校正外部测试终稿结果。

当前终稿统一口径：
- final aligned cohort: 1897 patients
- train/val/external_test: 1317/177/403
- external_test N: 403
- inputs: colposcopy + OCT + HPV/TCT/Age
- no reports, no VLM evidence cache

任务：

1. 新建目录：
   paper_revision/results/archive_legacy_mixed_n/

2. 扫描以下目录中的 CSV/JSON/MD：
   paper_revision/results/
   paper_revision/tables/
   paper_revision/results/tables/
   paper_revision/results/robustness/
   paper_revision/results/clinical_decision/
   paper_revision/results/coe_faithfulness/
   paper_revision/results/loco/

3. 对每个结果文件读取 n / split / method / run_id 字段。
   如果发现 external_test n 不是 403，或混有 148/196/283，则：
   - 复制到 archive_legacy_mixed_n/
   - 在原位置保留也可以，但必须写入 legacy_results_manifest.csv 标记为 legacy_not_for_main_paper

4. 新建：
   paper_revision/results/CLEAN_RERUN_SCOPE.md

写明：
   Only results generated from the corrected 1897-patient cohort with 403-case external test are eligible for main-paper tables. Older 148/196/283 external-test outputs are archived and must not be used in main manuscript tables.

5. 输出：
   paper_revision/results/archive_legacy_mixed_n/legacy_results_manifest.csv
   paper_revision/results/current_result_status_before_clean_rerun.csv

6. 不删除任何旧结果，只做归档和标记。
```

---

# Prompt R1. Verify corrected cohort, split, and feature cache before rerun

```text
请在正式重跑前执行校正队列与特征缓存一致性检查。

任务：

1. 新建脚本：
   paper_revision/scripts/verify_corrected_5center_cohort.py

2. 检查以下文件：
   paper_revision/splits/full_multimodal_resplit/final_1897_case_index.csv
   paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/train_labels.csv
   paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/val_labels.csv
   paper_revision/splits/target_adapted_validation/all_center_patient_holdout_70_10_20/external_test_labels.csv
   paper_revision/cache/patch_features_final_1897.pt
   paper_revision/configs/corrected_5center_elbo_structured_prior_config.py
   或当前真实使用的 corrected 5-center config

3. 强制断言：
   final case index = 1897
   train = 1317
   val = 177
   external_test = 403
   train + val + external_test = 1897
   feature cache contains all case keys
   no active report fields
   use_vlm_retriever=False
   no_report_mode=True
   clinical variables are HPV/TCT/Age only

4. 输出：
   paper_revision/results/real_50epoch_5center_corrected/cohort_cache_verification.json
   paper_revision/results/real_50epoch_5center_corrected/cohort_cache_verification.md

5. 如果任何断言失败，停止后续重跑。
```

---

# Prompt R2. Confirm Stage-1 clinical semantic adapter is actually loaded in Stage-2

```text
请确认 Stage-1 clinical semantic adapter checkpoint 是否真实接入 Stage-2 HyDRA-CoE 主模型训练。

背景：
stage1_clinical_semantic_adapter_best.pth 已存在，但需要确认主训练是否加载该 checkpoint，并且只加载 adapter 权重，不覆盖 classifier/reliability/posterior/ASCCP/CoE。

任务：

1. 检查并修改：
   models/bio_cot_v3_2.py
   training/train_bio_cot_v3.2.py
   paper_revision/configs/corrected_5center_elbo_structured_prior_config.py

2. 确保 config 中存在：
   load_clinical_semantic_adapter_path = "paper_revision/results/stage1_clinical_semantic_adapter/checkpoints/stage1_clinical_semantic_adapter_best.pth"
   freeze_clinical_semantic_adapter_at_start = True
   unfreeze_clinical_semantic_adapter_epoch = 5
   adapter_lr_multiplier = 0.1

3. Stage-2 加载时只允许加载：
   note_projector
   text_adapter
   clinical_feature_projector
   align_proj_img
   align_proj_text
   shared_align_proj

4. 明确禁止加载：
   classifier
   variational_reliability
   posterior_refiner
   asccp_prior
   coe_readout
   center_discriminator

5. 新增日志：
   loaded_stage1_adapter: true/false
   loaded_adapter_keys
   skipped_non_adapter_keys
   adapter_trainable_epoch_1
   adapter_trainable_after_unfreeze
   adapter_parameter_count

6. 新建 smoke test：
   paper_revision/scripts/check_stage1_adapter_injection.py

输出：
   paper_revision/results/stage1_adapter_injection_check.json
   paper_revision/results/stage1_adapter_injection_check.md
```

---

# Prompt R3. Real rerun of Full HyDRA-CoE on corrected 403 external test

```text
请对 Full HyDRA-CoE 在校正后的五中心 1897 队列上执行真实 50 epoch 多种子训练，不允许 dry_run。

统一口径：
- train/val/external_test = 1317/177/403
- inputs = colposcopy + OCT + HPV/TCT/Age
- no reports
- no VLM evidence cache
- internal validation locked threshold
- external_test N must be 403

任务：

1. 使用 config：
   paper_revision/configs/corrected_5center_elbo_structured_prior_config.py
   如果该文件不存在，则从 all_center_elbo_structured_prior_config.py 复制并创建。

2. 运行 seeds：
   42, 123, 456

3. 每个 seed：
   - 训练 50 epoch
   - 保存 best checkpoint
   - evaluate internal_validation
   - evaluate external_test
   - prediction CSV 必须含 403 行 external_test
   - 输出 reliability/guideline/coe/z 字段

4. 输出目录：
   paper_revision/results/real_50epoch_5center_corrected/full_hydra_coe/

5. 写入 manifest：
   paper_revision/results/real_50epoch_5center_corrected/full_hydra_coe/full_model_run_manifest.csv

字段：
   method
   seed
   config_path
   checkpoint_path
   internal_prediction_csv
   external_prediction_csv
   external_n
   best_epoch
   best_val_auc
   best_val_ece
   status
   start_time
   end_time
   error_message

6. 构建锁阈值表：
   locked_thresholds_by_run.csv
   external_test_metrics_locked_threshold.csv
   formatted_main_performance_full_model.csv

7. 如果 external_n != 403，则该 run 标记为 failed，不得进入 formatted table。
```

---

# Prompt R4. Rerun baselines under the same corrected 403 protocol

```text
请在相同 corrected 5-center 1897 队列和 403 external_test 下重跑所有主基线，避免与旧 n=148/196 结果混杂。

基线列表：
1. Clinical only: HPV/TCT/Age
2. Colposcopy only
3. OCT only
4. Image concat fusion
5. Late fusion
6. Gated fusion
7. Cross-attention fusion
8. Same-backbone direct fusion
9. HyDRA-CoE Full

任务：

1. 为缺失的 baseline 新建 config：
   corrected_5center_clinical_only_config.py
   corrected_5center_colposcopy_only_config.py
   corrected_5center_oct_only_config.py
   corrected_5center_direct_concat_config.py
   corrected_5center_direct_late_config.py
   corrected_5center_direct_gated_config.py
   corrected_5center_direct_cross_attention_config.py

2. 所有 config 必须：
   data_root 指向 1317/177/403 split
   expected_aligned_n=1897
   expected_external_n=403
   use_vlm_retriever=False
   no_report_mode=True
   feature_cache_path=patch_features_final_1897.pt

3. seeds：
   42, 123, 456

4. 每个 baseline：
   - 训练 50 epoch
   - evaluate internal_validation
   - evaluate external_test
   - external_test prediction CSV 必须 403 行

5. 输出：
   paper_revision/results/real_50epoch_5center_corrected/baselines/baseline_run_manifest.csv
   paper_revision/results/real_50epoch_5center_corrected/baselines/baseline_metrics_locked_threshold.csv
   paper_revision/results/real_50epoch_5center_corrected/baselines/formatted_main_baseline_table.csv

6. 明确区分：
   - end-to-end trained result
   - feature-space upper-bound result

如果保留 AUC=0.864 的 feature-space best result，必须单独写入：
   feature_space_upper_bound_table.csv
不得与真实 50 epoch checkpoint 主表混在一起。
```

---

# Prompt R5. Rerun requirement-level ablations under corrected 403 protocol

```text
请重跑所有 requirement-level ablation，统一使用 403 例校正外部测试，不允许使用旧 n=148/196/283 结果。

Ablation configs：
1. w/o Clinical Semantic Adapter
2. w/o Clinical Structured Prior
3. w/o HPV
4. w/o TCT
5. w/o Age
6. Image only
7. Clinical only
8. w/o Variational Reliability
9. w/o Center-aware Reliability
10. w/o Posterior Refinement
11. w/o Guideline Prototype
12. w/o Counterfactual Robustness

任务：

1. 所有 ablation config 必须继承 corrected_5center_elbo_structured_prior_config.py。
2. 每个 config 输出独立目录。
3. seeds：
   42, 123, 456
4. epoch：
   50
5. 每个 run：
   - 训练
   - evaluate internal_validation
   - evaluate external_test
   - external_n 必须为 403
   - 使用 internal validation locked threshold

6. 输出：
   paper_revision/results/real_50epoch_5center_corrected/ablations/ablation_run_manifest.csv
   paper_revision/results/real_50epoch_5center_corrected/ablations/ablation_metrics_locked_threshold.csv
   paper_revision/results/real_50epoch_5center_corrected/ablations/ablation_formatted_table.csv

7. ablation_run_manifest.csv 必须包含：
   method
   scientific_question
   seed
   config_path
   checkpoint_path
   internal_prediction_csv
   external_prediction_csv
   external_n
   status
   error_message

8. 如果 external_n != 403，不得进入 ablation_formatted_table.csv。
```

---

# Prompt R6. Rerun missing-modality and corruption robustness using corrected full checkpoint

```text
请使用 corrected 5-center 50-epoch Full HyDRA-CoE checkpoint 重跑 robustness，统一 external_test N=403。

输入：
   来自 full_hydra_coe/full_model_run_manifest.csv 的 best checkpoints
   seeds 42/123/456

Missing modality settings：
   full
   remove_oct
   remove_colposcopy
   remove_clinical_text
   random_one
   random_two

Input corruption settings：
   colpo_blur
   colpo_brightness
   colpo_occlusion
   oct_speckle
   oct_stripe
   oct_intensity
   oct_bscan_dropout

Severity：
   mild=1.0
   moderate=2.0
   severe=3.0

任务：

1. 所有 prediction CSV 的 external_n 必须为 403。
2. 所有指标使用 internal_validation locked threshold，不得在 corrupted external test 上重新调阈值。
3. clinical text 不参与 corruption，只参与 remove_clinical_text。
4. 输出 reliability shift：
   reliability_oct
   reliability_colposcopy
   reliability_clinical_text
   precision_oct
   precision_colposcopy
   precision_clinical_text

输出：
   paper_revision/results/real_50epoch_5center_corrected/robustness/missing_modality_robustness_metrics.csv
   paper_revision/results/real_50epoch_5center_corrected/robustness/missing_modality_degradation_vs_full.csv
   paper_revision/results/real_50epoch_5center_corrected/robustness/formatted_missing_modality_table.csv
   paper_revision/results/real_50epoch_5center_corrected/robustness/corruption_robustness_metrics.csv
   paper_revision/results/real_50epoch_5center_corrected/robustness/corruption_degradation_by_severity.csv
   paper_revision/results/real_50epoch_5center_corrected/robustness/formatted_corruption_table.csv
```

---

# Prompt R7. Rebuild label-noise stress test under corrected 403 protocol

```text
请基于 corrected 5-center 1897 队列重新执行 label-noise stress test。

规则：
- 只扰动训练集 1317 例中的标签。
- val 177 和 external_test 403 保持干净。
- clinical_info 和 clinical_features 不得改动。
- external_n 必须为 403。

Noise rates：
   0.05
   0.10
   0.20

Seeds：
   42, 123, 456

任务：
1. 重新生成 label-noise splits：
   paper_revision/splits/label_noise_corrected_5center/

2. 每个 noise rate × seed：
   - 训练 Full HyDRA-CoE 50 epoch
   - evaluate internal_validation
   - evaluate external_test
   - locked threshold from internal_validation

3. 输出：
   paper_revision/results/real_50epoch_5center_corrected/label_noise/label_noise_run_manifest.csv
   paper_revision/results/real_50epoch_5center_corrected/label_noise/label_noise_metrics.csv
   paper_revision/results/real_50epoch_5center_corrected/label_noise/label_noise_degradation_vs_clean.csv
   paper_revision/results/real_50epoch_5center_corrected/label_noise/formatted_label_noise_table.csv

4. 所有表中必须显示 external_n=403。
```

---

# Prompt R8. Rebuild LOCO and center-wise calibration with clean manifest

```text
请基于 final_1897_case_index.csv 重新执行 LOCO 和 center-wise calibration，生成清晰 manifest。

任务：

1. 使用 paper_revision/splits/loco/ 下五中心 split。
2. 对每个 held-out center：
   - train on other centers
   - validation from training centers only
   - test on held-out center only
   - threshold locked from validation only
   - held-out center 不参与模型选择

3. seeds：
   42, 123, 456

4. 每个 LOCO run 输出：
   train_n
   val_n
   heldout_test_n
   heldout_positive
   heldout_negative
   checkpoint_path
   prediction_csv
   locked_threshold
   AUC if both classes exist
   ECE
   Brier

5. 输出：
   paper_revision/results/real_50epoch_5center_corrected/loco/loco_run_manifest.csv
   paper_revision/results/real_50epoch_5center_corrected/loco/loco_metrics_by_center.csv
   paper_revision/results/real_50epoch_5center_corrected/loco/loco_formatted_table.csv
   paper_revision/results/real_50epoch_5center_corrected/loco/centerwise_ece_brier_table.csv
   paper_revision/results/real_50epoch_5center_corrected/loco/centerwise_reliability_summary.csv

6. 如果某中心只有单一类别：
   AUC=NA
   reason=single_class_heldout_center
```

---

# Prompt R9. Rebuild CoE faithfulness proxy using corrected full checkpoints

```text
请使用 corrected 5-center Full HyDRA-CoE 50-epoch checkpoints 重跑 CoE faithfulness proxy。

输入：
   full_hydra_coe/full_model_run_manifest.csv 中 status=success 的 checkpoints

External test：
   403 cases

实验：
1. Evidence removal：
   full
   remove clinical text
   remove colposcopy
   remove OCT

2. Clinical text swap：
   swap clinical_info + clinical_features across cases while keeping images fixed

3. Posterior trajectory：
   z0 -> z1 -> z2 -> z3 delta

4. Failure case mining：
   false negative with high OCT reliability
   false positive with high clinical reliability
   high-confidence wrong predictions
   large posterior conflict cases

输出：
   paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_evidence_removal_predictions.csv
   paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_clinical_text_swap_predictions.csv
   paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_posterior_trajectory_summary.csv
   paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_failure_cases.csv
   paper_revision/results/real_50epoch_5center_corrected/coe_faithfulness/coe_faithfulness_proxy_metrics.csv

必须在 README 或表注中写：
   These analyses are faithfulness proxies and are not equivalent to clinical validation of generated explanations.
```

---

# Prompt R10. Rebuild clinical decision quality from corrected 403 predictions

```text
请使用 corrected 5-center 403 external_test predictions 重建 clinical decision quality 表。

输入：
   full_hydra_coe predictions
   baseline predictions
   locked_thresholds_by_run.csv

要求：
1. 只使用 external_n=403 的 prediction CSV。
2. 使用 internal_validation locked threshold。
3. 不允许 external_test 重新选阈值。
4. 每个 method 输出 per 1000 women：
   predicted_positive_per_1000
   predicted_negative_per_1000
   true_positive_detected_per_1000
   false_positive_referral_per_1000
   false_negative_missed_per_1000
   low_yield_referral_per_1000
   number_needed_to_refer_for_one_positive
   net_benefit

5. Bootstrap 95% CI：
   n_bootstrap=1000
   stratify by y_true if possible
   fixed seed

输出：
   paper_revision/results/real_50epoch_5center_corrected/clinical_decision/clinical_decision_per_1000_table.csv
   paper_revision/results/real_50epoch_5center_corrected/clinical_decision/decision_curve_points.csv
   paper_revision/results/real_50epoch_5center_corrected/clinical_decision/formatted_clinical_utility_table.csv
```

---

# Prompt R11. Build final paper-ready tables from corrected-only results

```text
请基于 corrected 5-center 403 external_test 结果生成终稿主表和补充表，不得混入 n=148/196/283 旧结果。

输入目录：
   paper_revision/results/real_50epoch_5center_corrected/

任务：

1. 读取：
   full_hydra_coe/formatted_main_performance_full_model.csv
   baselines/formatted_main_baseline_table.csv
   ablations/ablation_formatted_table.csv
   robustness/formatted_missing_modality_table.csv
   robustness/formatted_corruption_table.csv
   label_noise/formatted_label_noise_table.csv
   loco/loco_formatted_table.csv
   clinical_decision/formatted_clinical_utility_table.csv

2. 强制检查：
   主性能表 external_n=403
   消融表 external_n=403
   鲁棒性 full baseline external_n=403
   clinical decision external_n=403

3. 输出 paper-ready tables：
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/Table1_main_performance.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/Table2_requirement_ablation.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/Table3_clinical_decision_per_1000.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/TableS1_cohort_image_volume.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/TableS2_robustness.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/TableS3_loco.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/TableS4_label_noise.csv
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/TableS5_coe_faithfulness_proxy.csv

4. 同时输出：
   paper_revision/results/real_50epoch_5center_corrected/paper_tables/PAPER_TABLE_SOURCE_AUDIT.md

该文件说明每张表来自哪个 prediction CSV、哪个 checkpoint、哪个 manifest。
```

---

# Prompt R12. Replace dry_run final manifest with real clean-rerun manifest

```text
请将 final_pipeline_manifest.csv 从 dry_run 状态更新为真实 clean rerun manifest。

任务：

1. 不要覆盖旧 dry_run manifest，先归档为：
   paper_revision/results/archive_legacy_mixed_n/final_pipeline_manifest_dry_run.csv

2. 新建真实 manifest：
   paper_revision/results/real_50epoch_5center_corrected/final_pipeline_manifest.csv

3. manifest 必须包含阶段：
   A_cohort_lock
   B_scope_cleanup
   C_clinical_mapping
   D_split_rebuild
   E_config_cache_verify
   F_stage1_adapter
   G_guideline_prototype
   H_full_model_training
   I_baseline_training
   J_ablation_training
   K_robustness
   L_label_noise
   M_loco
   N_coe_faithfulness
   O_clinical_decision
   P_paper_tables
   Q_naming_cleanup

4. 每行字段：
   stage
   method
   seed
   command
   config_path
   checkpoint_path
   prediction_csv
   output_table
   external_n
   status
   start_time
   end_time
   runtime_minutes
   error_message

5. 所有 dry_run=false。
6. 如果某阶段失败，status=failed，并记录错误，但不伪造成功。
```

---

# Prompt R13. Final paper-facing naming and method separation

```text
请做终稿前命名和结果口径清理。

任务：

1. 所有 paper-facing 表格中：
   Bio-COT -> HyDRA-CoE
   VLM evidence -> 删除或标为 not used
   report-guided -> 删除
   external n=148/196/283 -> legacy only, not for main paper

2. 主文 method 名称：
   HyDRA-CoE

3. 主文 full method display name：
   HyDRA-CoE: Guideline-conditioned Reliability Posterior Fusion for Multimodal Cervical Lesion Triage

4. 表格 method names：
   Clinical only
   Colposcopy only
   OCT only
   Image concat fusion
   Late fusion
   Gated fusion
   Cross-attention fusion
   HyDRA-CoE w/o Clinical Semantic Adapter
   HyDRA-CoE w/o Clinical Structured Prior
   HyDRA-CoE w/o Reliability Posterior
   HyDRA-CoE w/o Center-aware Reliability
   HyDRA-CoE w/o Guideline Prototype
   HyDRA-CoE Full

5. 输出：
   paper_revision/results/real_50epoch_5center_corrected/name_cleanup_audit.csv
   paper_revision/results/real_50epoch_5center_corrected/PAPER_READY_RESULT_SCOPE.md
```

---

## 3. 论文实验章节对应修改

### 4.1 Cohort construction and data alignment

使用：

```text
Prompt R0
Prompt R1
```

必须写入：

```text
3010 raw cases
1897 aligned tri-modal patients
137294 total images/B-scans
8216 colposcopy images
129078 OCT images/B-scans
train/val/test = 1317/177/403
no reports used
```

---

### 4.2 Implementation and evaluation protocol

使用：

```text
Prompt R2
Prompt R3
Prompt R12
Prompt R13
```

必须写入：

```text
Stage-1 clinical semantic adapter, if used, is loaded into Stage-2.
Threshold is locked on internal validation.
External test has 403 patients.
All results are from real runs, not dry-run manifests.
```

---

### 4.3 Main diagnostic performance

使用：

```text
Prompt R3
Prompt R4
Prompt R11
```

必须报告：

```text
Full HyDRA-CoE real 50-epoch external AUC
Baselines under identical 403-case protocol
Feature-space upper-bound separately, not mixed with end-to-end results
```

---

### 4.4 Requirement-level ablation

使用：

```text
Prompt R5
Prompt R11
```

必须说明每个 ablation 对应：

```text
semantic grounding
evidence reliability
guideline prototype
center nuisance robustness
```

---

### 4.5 Robustness under missing modality and corruption

使用：

```text
Prompt R6
Prompt R11
```

必须报告：

```text
external_n=403
locked threshold
reliability shift
performance degradation
```

---

### 4.6 Cross-center generalization and calibration

使用：

```text
Prompt R8
Prompt R11
```

必须报告：

```text
LOCO held-out center results
center-wise ECE/Brier
single-class center caveats if applicable
```

---

### 4.7 Label-noise stress test

使用：

```text
Prompt R7
Prompt R11
```

可放补充材料。

---

### 4.8 CoE faithfulness proxy

使用：

```text
Prompt R9
Prompt R11
```

必须加限定：

```text
faithfulness proxy, not clinical validation of generated explanations
```

---

### 4.9 Clinical decision quality

使用：

```text
Prompt R10
Prompt R11
```

必须报告：

```text
per-1000 women metrics
locked threshold
bootstrap 95% CI
```

---

## 4. 最小可发表实验集

如果时间有限，优先完成下面 6 项：

```text
1. R1: corrected cohort/cache verification
2. R3: Full HyDRA-CoE 3 seeds real 50-epoch rerun
3. R4: Main baselines under 403 protocol
4. R5: Core ablations under 403 protocol
5. R6: Missing/corruption robustness under 403 protocol
6. R10/R11: Clinical decision quality and paper-ready tables
```

LOCO、label-noise、CoE faithfulness 可以进入补充，但最好仍按 corrected 403/1897 口径重建。

---

## 5. 终稿结果使用规则

```text
Use in main paper:
  results/real_50epoch_5center_corrected/paper_tables/*

Use in supplement:
  corrected label-noise, LOCO, CoE faithfulness proxy

Do not use in main paper:
  any table with external_n=148
  any table with external_n=196
  any table with external_n=283
  final_pipeline_manifest.csv if all rows are dry_run
  mixed-n requirement_ablation_metrics.csv
  mixed-n robustness or clinical decision tables
```

---

## 6. 最终判断

当前项目已经完成“代码实现层面”的大部分工作，但终稿还需要完成“统一口径真实重跑层面”的收口。下一步唯一主线应是：

```text
Corrected 1897-patient cohort
→ 403-case external test
→ real 50-epoch checkpoints
→ locked threshold
→ corrected-only tables
→ paper-ready manifest
```

不要再新增方法模块。只做 clean rerun、clean tables、clean manuscript linkage。
