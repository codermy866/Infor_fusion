from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoAgeConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o Age"
    experiment_description: str = "Tests contribution of age within HPV/TCT/Age clinical variables."
    ablate_age: bool = True
    output_dir: str = "paper_revision/results/requirement_ablations/no_age/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_age/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_age/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
