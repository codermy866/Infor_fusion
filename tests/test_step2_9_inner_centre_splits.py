from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/publishable_v2/step2_9_domain_generalisation_recovery"


def test_outer_test_never_used_in_inner_validation_or_training():
    splits = pd.read_csv(OUT / "splits/inner_centre_validation_splits.csv")
    assert not splits.empty
    assert not (splits["center_name"] == splits["outer_test_center"]).any()
    assert set(splits["inner_role"]) == {"inner_train", "inner_validation"}


def test_inner_validation_has_centre_holdout():
    splits = pd.read_csv(OUT / "splits/inner_centre_validation_splits.csv")
    for (outer, inner), g in splits.groupby(["outer_test_center", "inner_validation_center"]):
        assert outer != inner
        val = g[g["inner_role"] == "inner_validation"]
        assert not val.empty
        assert set(val["center_name"]) == {inner}
        train = g[g["inner_role"] == "inner_train"]
        assert inner not in set(train["center_name"])
        assert outer not in set(train["center_name"])
