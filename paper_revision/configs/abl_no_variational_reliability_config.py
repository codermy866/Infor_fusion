from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoVariationalReliabilityConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o Reliability Posterior"
    experiment_description: str = "Tests contribution of variational modality reliability posterior."
    use_variational_reliability: bool = False
    fusion_strategy: str = "gated"
    output_dir: str = "paper_revision/results/requirement_ablations/no_variational_reliability/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_variational_reliability/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_variational_reliability/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
