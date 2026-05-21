from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoTCTConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o TCT"
    experiment_description: str = "Tests contribution of TCT variables within the no-report clinical text modality."
    ablate_tct: bool = True
    output_dir: str = "paper_revision/results/requirement_ablations/no_tct/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_tct/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_tct/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
