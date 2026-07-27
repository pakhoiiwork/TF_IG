# BL-PPG-Artifact-Removal

## Environment Setup

Follow these steps:
1. Run the setup script to initialize the environment:
   ```bash
   bash setup.sh
   ```

2. Create and activate the conda environment:
   ```bash
   conda activate artrmv
   pip install -r requirements.txt
   ```

3. (Optional) Verify the data handler setup:
   ```bash
   python test_handler.py
   ```

## Dataset Configuration

### Downloading Datasets

Before proceeding with training or inference, download and configure the required datasets as described in [datasets/README.md](datasets/README.md).

Currently supported datasets:
- **BIDMC PPG and Respiration Dataset** — PhysioNet
- **CapnoBase** — Borealis Data Repository

### Dataset Setup Instructions

Refer to [datasets/README.md](datasets/README.md) for detailed instructions on:
- Dataset sources and download links
- Directory structure and expected file formats
- Local restoration procedures


## Training

### Configuration

(Optional) Modify training parameters in `/configs/train_configs/`

### Running Training

Execute the training script with your desired configuration: `python main.py --config [path to config file]`. For example:

```bash
python main.py --config configs/train_configs/bidmc.yaml
```

### Output

All training artifacts are saved to `/pretrained_models/`, including:
- Training logs
- Model checkpoints
- Training history

### Monitoring Training Progress

View real-time metrics using TensorBoard:

```bash
tensorboard --logdir=pretrained_models/[model_name]/logs
```

Example for BIDMC-trained model:
```bash
tensorboard --logdir=pretrained_models/baseline_bidmc/logs
```

## Inference

### Configuration

(Optional) Modify inference parameters in `/configs/infer_configs/`

### Running Inference

Execute the inference script with your desired configuration: `python main.py --config [path to config file]`. For example:

```bash
python main.py --config configs/infer_configs/bidmc.yaml
```

### Output

Upon successful completion, evaluation plots comparing predicted and ground-truth heart rate are saved to `/pretrained_models/[model_name]/eval_predictions_plot.png`.

Example output path for BIDMC:
```
pretrained_models/baseline_bidmc/eval_predictions_plot.png
```

## Project Structure

```
BL-PPG-Artifact-Removal/
├── main.py                      # Entry point for training and inference
├── test_handler.py              # Data handler verification script
├── setup.sh                      # Environment initialization
├── requirements.txt             # Python dependencies
├── configs/
│   ├── train_configs/           # Training configuration files
│   └── infer_configs/           # Inference configuration files
├── pretrained_models/           # Training outputs and checkpoints
├── data_handler/                # Data loading and preprocessing
└── datasets/                    # Dataset storage and documentation
```

## Documentation

- [Dataset Setup Guide](datasets/README.md)
- Configuration examples in `/configs/`

## Support

For issues or questions, please refer to the project's issue tracker or contact the development team.