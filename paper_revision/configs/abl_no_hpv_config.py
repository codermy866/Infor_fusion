from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoHPVConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o HPV"
    experiment_description: str = "Tests contribution of HPV variables within the no-report clinical text modality."
    ablate_hpv: bool = True
    output_dir: str = "paper_revision/results/requirement_ablations/no_hpv/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_hpv/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_hpv/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
