from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_5_full_hydra_vlm_recovery"


def test_auxiliary_oct_inventory_exists():
    inv = OUT / "aux_oct_pretraining/audit/aux_oct_image_inventory.csv"
    assert inv.exists()
    df = pd.read_csv(inv)
    assert {"Hua_Xi", "XiangYa"}.issubset(set(df["source_center"].astype(str)))
    assert len(df) > 0

