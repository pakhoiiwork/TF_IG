from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neural_methods.trainers.baseline_mlp_trainer import BaselineMlpTrainer
from neural_methods.trainers.baseline_transformer_trainer import BaselineTransformerTrainer
from neural_methods.trainers.baseline_2branch_trainer import Baseline2BranchTrainer
from neural_methods.trainers.stag_hr_trainer import StagHrTrainer

# Mapping from CLI model names to (trainer class, config model key)
TRAINER_REGISTRY = {
    "base": (BaselineMlpTrainer, "base"),
    "base_trans": (BaselineTransformerTrainer, "base_trans"),
    "base_2branch": (Baseline2BranchTrainer, "base_2branch"),
    "base_i2branch": (StagHrTrainer, "base_i2branch"),
}


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config(config_path, model_name):
    """Load a structured YAML config and merge shared + model-specific params
    into a flat SimpleNamespace for backward compatibility with the trainers.

    The YAML has top-level keys: dataset_name, data_path, training, data, models.
    This function flattens `training.*`, `data.*`, and `models.<model_name>.*`
    into a single namespace alongside `dataset_name` and `data_path`.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    flat = {}

    # Top-level scalars
    flat["dataset_name"] = raw["dataset_name"]
    flat["data_path"] = raw["data_path"]

    # Merge training section
    if "training" in raw:
        flat.update(raw["training"])

    # Merge data section
    if "data" in raw:
        flat.update(raw["data"])

    # Merge model-specific section
    if "models" in raw and model_name in raw["models"]:
        flat.update(raw["models"][model_name])

    return to_namespace(flat)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate PPG → HR models")
    parser.add_argument("--model", required=True, choices=list(TRAINER_REGISTRY.keys()),
                        help="Model type to train")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path, model_name=args.model)

    trainer_cls, _ = TRAINER_REGISTRY[args.model]
    trainer = trainer_cls(config)
    trainer.run()


if __name__ == "__main__":
    main()