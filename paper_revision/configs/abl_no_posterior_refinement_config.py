from dataclasses import dataclass
from pathlib import Path
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class AblNoPosteriorRefinementConfig(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE w/o Posterior Refinement"
    experiment_description: str = "Tests contribution of sequential posterior refinement over clinical, colposcopy, and OCT evidence."
    use_posterior_refinement: bool = False
    output_dir: str = "paper_revision/results/requirement_ablations/no_posterior_refinement/results"
    checkpoint_dir: str = "paper_revision/results/requirement_ablations/no_posterior_refinement/checkpoints"
    log_dir: str = "paper_revision/results/requirement_ablations/no_posterior_refinement/logs"

    def __post_init__(self):
        super().__post_init__()
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
