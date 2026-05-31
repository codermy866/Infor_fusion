import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def test_full_model_required_modules_active():
    path = OUT / "audit/full_hydra_codepath_inventory.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    modules = {row["module"]: row["status"] for row in data["modules"]}
    assert modules["multimodal_cross_attention_transformer"] == "FOUND_AND_REUSABLE"
    assert modules["posterior_refinement"] == "FOUND_AND_REUSABLE"
    assert modules["asccp_prototype_prior"] == "FOUND_AND_REUSABLE"
    assert modules["coe_readout_trajectory"] == "FOUND_AND_REUSABLE"


def test_model_not_called_full_if_module_missing():
    path = OUT / "audit/full_model_forward_output_check.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["missing_or_inactive_required_modules"]:
        assert "Partial" in data["model_registry_name"]

