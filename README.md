## Environment Setup

Follow these steps:
1. Run the setup script to initialize the environment:
   ```bash
   bash setup.sh
   ```

2. Create and activate the conda environment:
   ```bash
   conda activate tfig
   pip install -r requirements.txt
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