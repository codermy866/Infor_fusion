from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_10_target_adaptation_final_if_decision"


def test_final_tables_exist_in_three_formats():
    for stem in [
        "Table1_Protocol_Separation",
        "Table2_Main_Target_Adaptation_Result",
        "Table3_Xiangyang_Rescue_Analysis",
        "Table4_Final_IF_Decision",
        "Table5_Final_Claim_Audit",
    ]:
        for ext in ["csv", "md", "tex"]:
            assert (OUT / f"tables/{stem}.{ext}").exists()


def test_final_figures_exist_with_source_data():
    stems = [
        "Figure1_Final_Evidence_Ladder",
        "Figure2_Target_Adaptation_Effect",
        "Figure3_Xiangyang_Failure_Rescue",
        "Figure4_Protocol_Transparency",
        "Figure5_Final_IF_Decision_Map",
    ]
    for stem in stems:
        for ext in ["pdf", "svg", "png"]:
            assert (OUT / f"figures/{stem}.{ext}").exists()
    for idx in range(1, 6):
        assert (OUT / f"figures/source/Figure{idx}_source.csv").exists()
