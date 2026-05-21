from dataclasses import dataclass
from paper_revision.configs.all_center_elbo_structured_prior_config import AllCenterELBOStructuredPriorConfig


@dataclass
class LabelNoise10Config(AllCenterELBOStructuredPriorConfig):
    experiment_name: str = "HyDRA-CoE Label Noise 0.10"
    noise_rate: float = 0.10
    data_root: str = "paper_revision/splits/label_noise/noise_0p10/seed42"
    output_dir: str = "paper_revision/results/label_noise/noise_0p10/results"
    checkpoint_dir: str = "paper_revision/results/label_noise/noise_0p10/checkpoints"
    log_dir: str = "paper_revision/results/label_noise/noise_0p10/logs"
