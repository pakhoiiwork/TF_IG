# Datasets

This repository depends on large raw datasets that are not committed to version control.
Place the datasets in this folder using the structure below so the data loaders and tests can find them without any code changes.

## Required Sources

### BIDMC PPG and Respiration Dataset

Source: https://physionet.org/content/bidmc/1.0.0/

The BIDMC dataset contains 53 eight-minute recordings derived from the MIMIC II matched waveform database. Each recording includes PPG, ECG, impedance respiration, breath annotations, and derived physiological numerics. The dataset is provided here in CSV form for use by the repository loaders.

The code in this repository expects the CSV export to be unpacked at:

```text
datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv/
```

Expected files for each subject:

```text
bidmc_##_Breaths.csv
bidmc_##_Numerics.csv
bidmc_##_Signals.csv
bidmc_##_Fix.txt
```

The original BIDMC package also contains WFDB files, a MATLAB bundle, checksum files, and dataset documentation. Keep the full extracted archive in place if you need those formats for analysis or validation.

### CapnoBase

Source: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT

CapnoBase is a benchmark dataset for respiratory-rate estimation from photoplethysmography. Each case is an eight-minute recording and the extracted CSV version used by this repository includes signal, labels, reference trends, metadata, parameters, and Smart Fusion results.

The code in this repository expects the CSV export to be unpacked at:

```text
datasets/CapnoBase/data/csv/
```

The companion preview images live under:

```text
datasets/CapnoBase/preview/
```

Common files in the CSV export follow this pattern:

```text
####_8min_signal.csv
####_8min_labels.csv
####_8min_reference.csv
####_8min_param.csv
####_8min_meta.csv
####_8min_SFresults.csv
```

## Folder Layout

```text
datasets/
├── README.md
├── bidmc-ppg-and-respiration-dataset-1.0.0/
│   ├── bidmc_csv/
│   ├── bidmc01.hea
│   ├── bidmc01.dat
│   ├── bidmc01n.hea
│   ├── bidmc01n.dat
│   ├── ...
│   ├── bidmc_data.mat
│   ├── RECORDS
│   ├── LICENSE
│   ├── SHA256SUMS.txt
│   └── README
├── CapnoBase/
│   ├── data/
│   │   └── csv/
│   ├── preview/
│   ├── MANIFEST.TXT
│   └── README.txt
└── IEEEE Signal Processing Cup-2015/
	├── TestData/
	├── TrueBPM/
	└── competition_data/
```

## How To Restore Locally

1. Download the BIDMC archive from PhysioNet and extract it under `datasets/bidmc-ppg-and-respiration-dataset-1.0.0/`.
2. Download the CapnoBase archive and extract it under `datasets/CapnoBase/`.
3. Verify that the CSV subfolders exist at the exact paths listed above.
4. Do not commit the raw dataset files to the repository; keep them as local artifacts only.

## Notes For Collaborators

- The loaders in `data_handler/bidmc_hanlder.py` and `data_handler/capnobase_handler.py` assume the directory names shown above.
- If you change the dataset layout, update the loader paths and the test paths in `test_handlers.py` at the same time.
- The repository also contains an archived `IEEEE Signal Processing Cup-2015/` folder, which is separate from the BIDMC and CapnoBase pipelines.