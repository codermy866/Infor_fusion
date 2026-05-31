#!/usr/bin/env python3
"""After full LOCO ablation: identify harmful/neutral modules and prune defaults."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts.shared_lora_biocot.run_1897_improved_loco import (  # noqa: E402
    ABLATION_PRESETS,
    G1_CONTROL,
)

OUT_ROOT = EXP_ROOT / "outputs/publishable_v2/shared_lora_biocot/improved_1897"
ABLATION_ROOT = OUT_ROOT / "ablations"
CONTROL_GROUP = "g1"
KEEP_MARGIN = 0.010  # drop if removal hurts less than this vs control
NEUTRAL_MARGIN = 0.005

# leave-one-out group -> config overrides when component is pruned
PRUNE_OVERRIDES: dict[str, dict[str, Any]] = {
    "no_align": {"lambda_align": 0.0},
    "no_ot": {"use_ot": False, "lambda_ot": 0.0},
    "no_bridge": {"lambda_colpo_bridge_ot": 0.0, "lambda_colpo_bridge_align": 0.0},
    "no_coe": {"lambda_coe": 0.0, "use_coe_readout": False, "use_coe_supervision": False},
    "g3": {"lambda_coe": 0.0, "use_coe_readout": False, "use_coe_supervision": False},
    "no_consist": {"use_dual": False, "lambda_consist": 0.0},
    "no_posterior": {"use_posterior_refinement": False, "lambda_posterior_smooth": 0.0},
    "no_variational": {"use_variational_reliability": False, "lambda_reliability_kl": 0.0},
    "no_modality_likelihood": {"use_modality_likelihood": False, "lambda_modality_likelihood": 0.0},
    "no_cross_attn": {"use_cross_attn": False},
    "no_bridge_module": {
        "use_colpo_lora_bridge": False,
        "lambda_colpo_bridge_ot": 0.0,
        "lambda_colpo_bridge_align": 0.0,
    },
    "no_center_reliability": {"use_center_aware_reliability": False},
}

ALWAYS_PRUNED = {
    "use_adversarial": False,
    "lambda_adv": 0.0,
}

CONFIG_FILES = [
    EXP_ROOT / "config.py",
    EXP_ROOT / "configs/shared_lora_loco_template.py",
]
PRUNED_CONFIG_PY = EXP_ROOT / "configs/ablation_pruned_loco_config.py"


def _write_pruned_loco_config(pruned: dict[str, Any], meta: dict[str, Any]) -> None:
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated production config after full 1897 LOCO ablation."""',
        "from dataclasses import dataclass",
        "from configs.shared_lora_loco_template import SharedLoRALOCOConfig",
        "",
        "",
        "@dataclass",
        "class AblationPrunedLOCOConfig(SharedLoRALOCOConfig):",
        f'    """Best stack: {meta.get("best_group", "g1")} (ablation AUC={meta.get("best_auc", 0):.4f})."""',
        "",
    ]
    for key, value in sorted(pruned.items()):
        if key in {"description", "removed_components"}:
            continue
        if isinstance(value, bool):
            lines.append(f"    {key}: bool = {value}")
        elif isinstance(value, int):
            lines.append(f"    {key}: int = {value}")
        elif isinstance(value, float):
            lines.append(f"    {key}: float = {value}")
    lines.append("")
    PRUNED_CONFIG_PY.write_text("\n".join(lines), encoding="utf-8")


def load_auc(group: str) -> float | None:
    path = ABLATION_ROOT / group / "tables/Table_Improved1897_LOCO_Fold_Metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    vals = df["cin2plus_auc"].dropna()
    return float(vals.mean()) if len(vals) else None


def decide_pruning(comparison_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    control_auc = next((r["mean_cin2plus_auc"] for r in comparison_rows if r["group"] == CONTROL_GROUP), None)
    if control_auc is None:
        raise RuntimeError(f"Control group {CONTROL_GROUP} not complete; cannot prune.")

    pruned = dict(G1_CONTROL)
    pruned.update(ALWAYS_PRUNED)
    decisions = []

    for group, overrides in PRUNE_OVERRIDES.items():
        auc = next((r["mean_cin2plus_auc"] for r in comparison_rows if r["group"] == group), None)
        if auc is None:
            decisions.append(
                {
                    "group": group,
                    "component": ABLATION_PRESETS.get(group, {}).get("removed_components", group),
                    "status": "pending",
                    "auc": None,
                    "delta_vs_g1": None,
                    "action": "wait",
                }
            )
            continue
        delta = float(auc) - float(control_auc)
        if delta >= -NEUTRAL_MARGIN:
            action = "remove"
            pruned.update(overrides)
            reason = "removal neutral or improves AUC → prune"
        elif delta < -KEEP_MARGIN:
            action = "keep"
            reason = "removal clearly hurts AUC → keep"
        else:
            action = "remove"
            pruned.update(overrides)
            reason = "small benefit, prune for simplicity"
        decisions.append(
            {
                "group": group,
                "component": ABLATION_PRESETS.get(group, {}).get("removed_components", group),
                "status": "done",
                "auc": auc,
                "delta_vs_g1": delta,
                "action": action,
                "reason": reason,
            }
        )

    # If minimal g2 beats control, adopt g2 as production stack
    g2_auc = next((r["mean_cin2plus_auc"] for r in comparison_rows if r["group"] == "g2"), None)
    best_group = CONTROL_GROUP
    best_auc = control_auc
    if g2_auc is not None and g2_auc > control_auc + NEUTRAL_MARGIN:
        pruned = dict(ABLATION_PRESETS["g2"])
        pruned.update(ALWAYS_PRUNED)
        best_group = "g2"
        best_auc = g2_auc

    for row in comparison_rows:
        if row["group"] in {"g2", "g4", "best", "g0"} and row.get("mean_cin2plus_auc") is not None:
            auc = float(row["mean_cin2plus_auc"])
            if auc > best_auc + NEUTRAL_MARGIN:
                best_auc = auc
                best_group = row["group"]
                if row["group"] in ABLATION_PRESETS:
                    pruned = dict(ABLATION_PRESETS[row["group"]])
                    pruned.update(ALWAYS_PRUNED)

    decision_df = pd.DataFrame(decisions)
    meta = {"best_group": best_group, "best_auc": best_auc, "control_auc": control_auc}
    return pruned, decision_df, meta


def apply_to_config_file(path: Path, pruned: dict[str, Any]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    changed = []
    for key, value in pruned.items():
        if key in {"description", "removed_components", "shared_lora_rank", "shared_lora_alpha"}:
            continue
        if isinstance(value, bool):
            pattern = rf"^(\s*{re.escape(key)}:\s*bool\s*=\s*)(?:True|False)"
            repl = rf"\g<1>{value}"
        elif isinstance(value, int):
            pattern = rf"^(\s*{re.escape(key)}:\s*int\s*=\s*)([-+]?\d+)"
            repl = rf"\g<1>{value}"
        elif isinstance(value, float):
            pattern = rf"^(\s*{re.escape(key)}:\s*float\s*=\s*)([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
            repl = rf"\g<1>{value}"
        else:
            continue
        new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if n:
            text = new_text
            changed.append(key)
    path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write pruned defaults into config.py files.")
    parser.add_argument("--control", default=CONTROL_GROUP)
    args = parser.parse_args()

    table_dir = OUT_ROOT / "tables"
    report_dir = OUT_ROOT / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in sorted(ABLATION_PRESETS):
        auc = load_auc(group)
        if auc is None:
            continue
        preset = ABLATION_PRESETS[group]
        rows.append(
            {
                "group": group,
                "description": preset.get("description", ""),
                "removed_components": preset.get("removed_components", ""),
                "mean_cin2plus_auc": auc,
            }
        )

    if not any(r["group"] == args.control for r in rows):
        print(f"Control {args.control} not ready. Completed: {[r['group'] for r in rows]}")
        sys.exit(1)

    pruned, decisions, meta = decide_pruning(rows)
    decisions.to_csv(table_dir / "Table_Ablation_Prune_Decisions.csv", index=False, encoding="utf-8-sig")
    (table_dir / "Table_Pruned_Production_Defaults.json").write_text(
        json.dumps(pruned, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_pruned_loco_config(pruned, meta)

    lines = [
        "# 消融后生产配置（自动裁剪）",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Control ({args.control}) AUC: **{meta['control_auc']:.4f}**",
        f"Selected production stack: **{meta['best_group']}** (AUC **{meta['best_auc']:.4f}**)",
        "",
        "## 裁剪决策",
        "",
        decisions.to_markdown(index=False),
        "",
        "## 生产默认配置",
        "",
        "```json",
        json.dumps(pruned, indent=2, ensure_ascii=False),
        "```",
    ]
    report_path = report_dir / "Report_Ablation_Pruned_Production.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.apply:
        for path in CONFIG_FILES:
            changed = apply_to_config_file(path, pruned)
            print(f"Updated {path.name}: {changed}")
    else:
        print("Dry-run only. Pass --apply to write config defaults.")

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
