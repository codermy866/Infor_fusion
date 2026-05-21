#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Config classes for corrected 403 external-test clean reruns (R3–R5)."""

from dataclasses import dataclass
from pathlib import Path

from paper_revision.configs.corrected_5center_elbo_structured_prior_config import (
    CorrectedFiveCenterELBOStructuredPriorConfig,
)
from paper_revision.configs.corrected403_base_config import CORRECTED_ROOT, STAGE1_ADAPTER

ROOT = Path(__file__).resolve().parents[2]


def _dirs(sub: str) -> tuple[str, str, str]:
    base = CORRECTED_ROOT / sub
    return (
        str(base / "results"),
        str(base / "checkpoints"),
        str(base / "logs"),
    )


@dataclass
class Corrected403FullHyDRAConfig(CorrectedFiveCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE Full"


@dataclass
class Corrected403ClinicalOnlyConfig(CorrectedFiveCenterELBOStructuredPriorConfig):
    experiment_name: str = "Clinical only"
    experiment_description: str = "HPV/TCT/Age clinical variables only."
    clinical_only_mode: bool = True
    load_clinical_semantic_adapter_path: str = str(STAGE1_ADAPTER)

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/clinical_only")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403ColposcopyOnlyConfig(CorrectedFiveCenterELBOStructuredPriorConfig):
    experiment_name: str = "Colposcopy only"
    colposcopy_only_mode: bool = True
    load_clinical_semantic_adapter_path: str = None

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/colposcopy_only")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403OctOnlyConfig(CorrectedFiveCenterELBOStructuredPriorConfig):
    experiment_name: str = "OCT only"
    oct_only_mode: bool = True
    load_clinical_semantic_adapter_path: str = None

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/oct_only")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403DirectFusionBase(CorrectedFiveCenterELBOStructuredPriorConfig):
    use_variational_reliability: bool = False
    use_posterior_refinement: bool = False
    use_asccp_prior: bool = False
    use_coe_readout: bool = False
    use_coe_supervision: bool = False
    use_dual: bool = False
    use_ot: bool = False
    direct_fusion_only: bool = True
    load_clinical_semantic_adapter_path: str = str(STAGE1_ADAPTER)


@dataclass
class Corrected403ConcatFusionConfig(Corrected403DirectFusionBase):
    experiment_name: str = "Image concat fusion"
    fusion_strategy: str = "concat"

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/concat_fusion")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403LateFusionConfig(Corrected403DirectFusionBase):
    experiment_name: str = "Late fusion"
    fusion_strategy: str = "late"

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/late_fusion")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403GatedFusionConfig(Corrected403DirectFusionBase):
    experiment_name: str = "Gated fusion"
    fusion_strategy: str = "gated"
    use_adaptive_gating: bool = True

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/gated_fusion")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


@dataclass
class Corrected403CrossAttentionFusionConfig(Corrected403DirectFusionBase):
    experiment_name: str = "Cross-attention fusion"
    fusion_strategy: str = "cross_attention"
    use_cross_attn: bool = True

    def __post_init__(self):
        out, ckpt, log = _dirs("baselines/cross_attention_fusion")
        self.output_dir, self.checkpoint_dir, self.log_dir = out, ckpt, log
        super().__post_init__()


ABLATION_SPECS = [
    ("no_clinical_semantic_adapter", "HyDRA-CoE w/o Clinical Semantic Adapter", {"load_clinical_semantic_adapter_path": None}),
    ("no_clinical_structured_prior", "HyDRA-CoE w/o Clinical Structured Prior", {"use_asccp_prior": False}),
    ("no_hpv", "HyDRA-CoE w/o HPV", {"disable_hpv": True}),
    ("no_tct", "HyDRA-CoE w/o TCT", {"disable_tct": True}),
    ("no_age", "HyDRA-CoE w/o Age", {"disable_age": True}),
    ("image_only", "HyDRA-CoE Image Only", {"image_only_mode": True}),
    ("clinical_only", "HyDRA-CoE Clinical Only", {"clinical_only_mode": True}),
    ("no_variational_reliability", "HyDRA-CoE w/o Reliability Posterior", {"use_variational_reliability": False, "fusion_strategy": "gated"}),
    ("no_center_aware_reliability", "HyDRA-CoE w/o Center-aware Reliability", {"use_center_aware_reliability": False}),
    ("no_posterior_refinement", "HyDRA-CoE w/o Posterior Refinement", {"use_posterior_refinement": False}),
    ("no_guideline_prototype", "HyDRA-CoE w/o Guideline Prototype", {"use_asccp_prior": False}),
    ("no_counterfactual_robustness", "HyDRA-CoE w/o Counterfactual Robustness", {"use_counterfactual_robustness": False}),
]


def write_ablation_config_files() -> list[Path]:
    """Emit per-ablation config .py files under corrected403_generated/."""
    gen_dir = ROOT / "paper_revision" / "configs" / "corrected403_generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for slug, display_name, overrides in ABLATION_SPECS:
        out, ckpt, log = _dirs(f"ablations/{slug}")
        lines = [
            "#!/usr/bin/env python",
            "# -*- coding: utf-8",
            "from dataclasses import dataclass",
            "from pathlib import Path",
            "from paper_revision.configs.corrected_5center_elbo_structured_prior_config import CorrectedFiveCenterELBOStructuredPriorConfig",
            "",
            "@dataclass",
            f"class Corrected403Abl_{slug}(CorrectedFiveCenterELBOStructuredPriorConfig):",
            f'    experiment_name: str = "{display_name}"',
        ]
        for key, value in overrides.items():
            if value is None:
                lines.append(f"    {key}: str = None")
            elif isinstance(value, bool):
                lines.append(f"    {key}: bool = {value}")
            elif isinstance(value, str):
                lines.append(f'    {key}: str = "{value}"')
            else:
                lines.append(f"    {key} = {value!r}")
        lines.extend(
            [
                f'    output_dir: str = "{out}"',
                f'    checkpoint_dir: str = "{ckpt}"',
                f'    log_dir: str = "{log}"',
                "",
                "    def __post_init__(self):",
                "        super().__post_init__()",
            ]
        )
        path = gen_dir / f"abl_{slug}.py"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


BASELINE_CONFIGS = {
    "clinical_only": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403ClinicalOnlyConfig",
    "colposcopy_only": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403ColposcopyOnlyConfig",
    "oct_only": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403OctOnlyConfig",
    "concat_fusion": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403ConcatFusionConfig",
    "late_fusion": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403LateFusionConfig",
    "gated_fusion": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403GatedFusionConfig",
    "cross_attention_fusion": "paper_revision/configs/corrected403_experiment_configs.py:Corrected403CrossAttentionFusionConfig",
}
