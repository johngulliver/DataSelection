from pathlib import Path

from DataSelection.configs.config_node import ConfigNode

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
CIFAR10_ROOT_DIR = Path("~/.torch/datasets/CIFAR10")

EXPERIMENT_DIR = PROJECT_ROOT_DIR / "logs"
BENCHMARK1_DIR = EXPERIMENT_DIR / "benchmark_1"
BENCHMARK2_DIR = EXPERIMENT_DIR / "benchmark_2"
DATA_CURATION_DIR = EXPERIMENT_DIR / "data_curation"


def get_train_output_dir(config: ConfigNode) -> str:
    """
    Returns default path to training checkpoint/tf-events output directory
    """
    config_output_dir = config.train.output_dir
    train_output_dir = EXPERIMENT_DIR / config_output_dir / f'seed_{config.train.seed:d}'
    return str(train_output_dir)
