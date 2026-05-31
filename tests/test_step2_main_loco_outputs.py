from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publishable_v2" / "step2_main_loco"
MANDATORY = {
    "ClinicalOnly_Logistic",
    "ClinicalOnly_XGBoost",
    "ColposcopyOnly_ViT",
    "OCTOnly_ViT",
    "ColposcopyOCT_EarlyConcat",
    "ColposcopyOCT_LateFusion",
    "ColposcopyOCTText_CrossAttention",
    "BioMedCLIP_Finetuned",
    "HyDRA_CoE_Full",
}


def test_predictions_use_all_1897_cases_once_per_model_seed_in_pooled_loco():
    pred = pd.read_csv(OUT / "predictions" / "patient_level_predictions_all_models.csv")
    assert set(pred["model_name"]) == MANDATORY
    for (model, seed), g in pred.groupby(["model_name", "seed"]):
        assert len(g) == 1897, (model, seed)
        assert g["case_id"].is_unique, (model, seed)


def test_table2_contains_mandatory_models():
    table = pd.read_csv(OUT / "tables" / "Table2_Main_LOCO_Diagnostic_Performance.csv")
    assert set(table["Method"]) == MANDATORY


def test_cin3plus_is_subset_of_cin2plus():
    pred = pd.read_csv(OUT / "predictions" / "patient_level_predictions_all_models.csv")
    assert ((pred["pathology_cin3plus"] == 1) <= (pred["pathology_cin2plus"] == 1)).all()
