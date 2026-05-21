from dataclasses import dataclass
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class LabelNoise05Config(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE Label Noise 0.05"
    noise_rate: float = 0.05
    data_root: str = "paper_revision/splits/label_noise/noise_0p05/seed42"
    output_dir: str = "paper_revision/results/label_noise/noise_0p05/results"
    checkpoint_dir: str = "paper_revision/results/label_noise/noise_0p05/checkpoints"
    log_dir: str = "paper_revision/results/label_noise/noise_0p05/logs"
