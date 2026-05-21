from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoGuidelinePrototypeConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o Guideline Prototype"
    experiment_description: str = "Tests contribution of HPV/TCT/Age guideline prototype prior."
    use_asccp_prior: bool = False
    lambda_asccp_ot: float = 0.0
    output_dir: str = "paper_revision/results/requirement_ablations/no_guideline_prototype/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_guideline_prototype/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_guideline_prototype/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
