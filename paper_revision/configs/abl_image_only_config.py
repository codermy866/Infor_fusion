from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblImageOnlyConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE Image Only"
    experiment_description: str = "Uses OCT and colposcopy only to quantify value of HPV/TCT/Age clinical variables."
    image_only_mode: bool = True
    output_dir: str = "paper_revision/results/requirement_ablations/image_only/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/image_only/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/image_only/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
