# Protocol Lock

Primary analysis is strict five-fold leave-one-center-out (LOCO). For each fold, the held-out center is used only for testing.

Validation thresholds are selected from one deterministic hard validation center among the remaining source centers. Target labels are not used for training or threshold selection.

Target-label-free logit median matching, when used, is transductive calibration because it uses the unlabeled target prediction distribution.

A fixed external split exists and is treated as optional supplementary material only; it is not mixed into the primary LOCO tables.

## Center Class Check
| center | n | cin2_classes | cin3_classes | one_class_cin2 | one_class_cin3 |
| --- | --- | --- | --- | --- | --- |
| 十堰市人民医院 | 496 | 0,1 | 0,1 | False | False |
| 恩施州中心医院 | 406 | 0,1 | 0,1 | False | False |
| 武大人民医院 | 89 | 1 | 0,1 | True | False |
| 荆州市第一人民医院 | 406 | 0,1 | 0,1 | False | False |
| 襄阳市中心医院 | 500 | 0,1 | 0,1 | False | False |
