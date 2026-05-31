from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_final_tables_exist_in_three_formats():
    for stem in [
        "Table1_Domain_Shift_Diagnosis",
        "Table2_Validation_Generalisation_Gap",
        "Table3_Domain_Generalisation_Recovery",
        "Table4_DG_Module_Contribution",
        "Table5_IF_Go_NoGo_Decision",
    ]:
        for ext in ["csv", "md", "tex"]:
            assert (OUT / f"tables/{stem}.{ext}").exists()


def test_figures_have_source_csv():
    for stem in [
        "Figure1_Centre_Domain_Shift_Diagnosis",
        "Figure2_Validation_Overselection_Diagnosis",
        "Figure3_Domain_Generalisation_Recovery",
        "Figure4_DG_Method_Contribution",
        "Figure5_Information_Fusion_Decision_Map",
    ]:
        for ext in ["pdf", "svg", "png"]:
            assert (OUT / f"figures/{stem}.{ext}").exists()
    for idx in range(1, 6):
        assert (OUT / f"figures/source/Figure{idx}_source.csv").exists()
