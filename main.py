from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from neural_methods.trainers.baseline_trainer import BaselineTrainer
from neural_methods.trainers.baseline_transformer_trainer import BaselineTransformerTrainer
from neural_methods.trainers.baseline_2branch_trainer import Baseline2BranchTrainer

def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return to_namespace(yaml.safe_load(handle))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline PPG artifact-removal model")
    parser.add_argument("--model", required=True, help="Model type to train")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    # save_path = ROOT / "baseline_model.pth"
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path)
    
    if args.model == "base_trans":
        trainer = BaselineTransformerTrainer(config)
    elif args.model == "base":
        trainer = BaselineTrainer(config)
    elif args.model == "base_2branch":
        trainer = Baseline2BranchTrainer(config)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    trainer.run()
    # trainer.save_model(save_path)


if __name__ == "__main__":
    main()