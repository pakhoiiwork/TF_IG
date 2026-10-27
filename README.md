# TF-IG

![Model Architecture](/images/TF_IG_Architect.png)

## Setup

Run these commands from the project folder:

```bash
bash setup.sh
conda activate tfig
pip install -r requirements.txt
```

Download and place the datasets in `datasets/` by following [datasets/README.md](datasets/README.md).

Expected dataset paths:

```text
datasets/
├── bidmc-ppg-and-respiration-dataset-1.0.0/
│   └── bidmc_csv/
└── CapnoBase/
    └── data/csv/
```

## Available models

| Model name | Implementation | Description |
|---|---|---|
| `base` | `neural_methods/models/baseline_mlp.py` | MLP baseline |
| `base_trans` | `neural_methods/models/baseline_transformer.py` | Patch-based Transformer baseline |
| `tf_ig` (main model) | `neural_methods/models/tf_ig.py` | TF-IG time/frequency dual-branch model |

## Train a model

Run one of the following commands from the project folder. Each command selects a model and a dataset configuration:

```bash
python main.py --model base --config configs/train_configs/bidmc.yaml
python main.py --model base_trans --config configs/train_configs/capnobase.yaml
python main.py --model tf_ig --config configs/train_configs/bidmc.yaml
```

The training configuration combines shared training settings, dataset settings, and the model settings selected by `--model`.

For a new experiment, update the relevant files in `configs/models/`, `configs/datasets/`, or `configs/common/`. The existing training and testing files continue to work as shown above.

## Test a trained model

After training, set `mode: "only_test"` and the checkpoint path in the matching file under `configs/infer_configs/`. Then run:

```bash
python main.py --model base --config configs/infer_configs/base/bidmc.yaml
python main.py --model base_trans --config configs/infer_configs/base_trans/capnobase.yaml
python main.py --model tf_ig --config configs/infer_configs/tf_ig/bidmc.yaml
```

Before testing, make sure that you have:

- a trained checkpoint at `model_path`;
- the split file created during training;
- the same dataset and window settings used during training.

Evaluation writes metrics, split metadata, preprocessing metadata, and prediction plots under the model output directory.

## Test on another dataset

To test on a different dataset, use the example configuration below. The training dataset is used for training and validation; the second dataset is used only for testing.

Example:

```bash
python main.py \
  --model tf_ig \
  --config configs/experiments/cross_bidmc_to_capnobase_tf_ig.yaml
```

The cross-dataset configuration declares:

```yaml
evaluation_protocol: cross_dataset
test_dataset_name: capnobase
test_data_path: datasets/CapnoBase/data/csv
```

The test dataset must provide compatible signals and targets. Input windows are resized automatically before they are passed to the model.

## Configuration layout

```text
configs/
├── common/
│   ├── runtime.yaml
│   └── training.yaml
├── datasets/
│   ├── bidmc.yaml
│   └── capnobase.yaml
├── models/
│   ├── base.yaml
│   ├── base_trans.yaml
│   └── tf_ig.yaml
├── experiments/
│   └── cross_bidmc_to_capnobase_tf_ig.yaml
├── train_configs/
└── infer_configs/
```

What each folder contains:

- `common/`: shared training and runtime defaults.
- `datasets/`: dataset path and window defaults.
- `models/`: architecture-specific parameters.
- `experiments/`: examples for testing across datasets.
- `train_configs/` and `infer_configs/`: ready-to-run training and testing settings.

## Project structure

```text
TF-IG/
├── main.py
├── configs/
├── data_handler/
│   ├── contracts.py
│   ├── registry.py
│   ├── splitters.py
│   ├── bidmc_hanlder.py
│   └── capnobase_handler.py
├── neural_methods/
│   ├── models/
│   │   ├── baseline_mlp.py
│   │   ├── baseline_transformer.py
│   │   └── tf_ig.py
│   └── trainers/
│       ├── base_trainer.py
│       ├── baseline_mlp_trainer.py
│       ├── baseline_transformer_trainer.py
│       └── tf_ig_trainer.py
├── utils/
│   ├── plotting.py
│   ├── preprocessing_config.py
│   ├── bandpass_filter.py
│   └── summary.py
├── datasets/
├── review.md
└── plan.md
```

## Outputs

Training and testing artifacts are written under:

```text
pretrained_models/
└── <model>_<dataset>/
    ├── best_model.pth
    ├── eval_predictions_test.png
    ├── logs/
    └── reports/
        ├── history.json
        ├── preprocessing_config.json
        ├── split_info.json
        └── test_report.json
```

Raw datasets and generated checkpoints should not be committed.

## Check model size

Use `utils/summary.py` to view a model's size and resource estimates:

```bash
python utils/summary.py --model tf_ig --device cpu
python utils/summary.py --model base_trans --sequence-length 1875 --device cpu
```
