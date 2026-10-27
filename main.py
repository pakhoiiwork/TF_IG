from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neural_methods.trainers.baseline_mlp_trainer import BaselineMlpTrainer
from neural_methods.trainers.baseline_transformer_trainer import BaselineTransformerTrainer
from neural_methods.trainers.tf_ig_trainer import TF_IGTrainer

# Mapping from CLI model names to (trainer class, config model key)
TRAINER_REGISTRY = {
    "base": (BaselineMlpTrainer, "base"),
    "base_trans": (BaselineTransformerTrainer, "base_trans"),
    "tf_ig": (TF_IGTrainer, "tf_ig"),
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

    The YAML has top-level keys: dataset_name, data_path, training, data, evaluation, models.
    This function flattens `training.*`, `data.*`, and `models.<model_name>.*`
    into a single namespace alongside `dataset_name` and `data_path`.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    flat = {}

    training_config = raw.get("training_config", "configs/common/training.yaml")
    shared_training_path = ROOT / training_config
    if shared_training_path.exists():
        with shared_training_path.open("r", encoding="utf-8") as handle:
            shared_training = yaml.safe_load(handle) or {}
        flat.update(shared_training)

    dataset_config = raw.get("dataset_config")
    if dataset_config:
        dataset_config_path = ROOT / dataset_config
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        with dataset_config_path.open("r", encoding="utf-8") as handle:
            flat.update(yaml.safe_load(handle) or {})

    required_keys = {"dataset_name", "data_path"}
    missing_keys = required_keys - flat.keys() - raw.keys()
    if missing_keys:
        raise ValueError(
            f"Configuration {config_path} is missing required keys: "
            f"{', '.join(sorted(missing_keys))}"
        )

    flat.update({
        key: raw[key]
        for key in ("dataset_name", "data_path")
        if key in raw
    })

    # Merge training section
    if "training" in raw:
        flat.update(raw["training"])

    # Merge data section
    if "data" in raw:
        flat.update(raw["data"])

    if "evaluation" in raw:
        flat.update(raw["evaluation"])

    # Merge model-specific section
    model_config_path = ROOT / raw.get(
        "model_config", f"configs/models/{model_name}.yaml"
    )
    if model_config_path.exists():
        with model_config_path.open("r", encoding="utf-8") as handle:
            flat.update(yaml.safe_load(handle) or {})

    model_configs = raw.get("models", {})
    if model_configs:
        if model_name not in model_configs:
            raise ValueError(
                f"Configuration {config_path} has no model section for {model_name!r}"
            )
        flat.update(model_configs[model_name])

    for path_key in ("data_path", "test_data_path", "model_path"):
        value = flat.get(path_key)
        if value and not Path(value).is_absolute():
            flat[path_key] = str((ROOT / value).resolve())

    flat["model_name"] = model_name
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