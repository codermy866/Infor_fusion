from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoClinicalSemanticAdapterConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o Clinical Semantic Adapter"
    experiment_description: str = "Ablates Stage-1 HPV/TCT/Age clinical semantic adapter transfer."
    load_clinical_semantic_adapter_path: str = None
    output_dir: str = "paper_revision/results/requirement_ablations/no_clinical_semantic_adapter/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_clinical_semantic_adapter/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_clinical_semantic_adapter/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
