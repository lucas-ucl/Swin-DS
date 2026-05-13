# Lacune Segmentation ICASSP Experiments

This repository contains the training, inference, and qualitative visualization code for the ICASSP lacune segmentation experiments. The original Colab notebooks have been converted into Python scripts for easier review, reuse, and version control.


## Setup

Create a fresh Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA driver before installing the remaining dependencies. The original Colab experiments used PyTorch 2.6 and MONAI 1.5.

## Data Layout

The scripts expect a Task3-style dataset containing subject folders named `sub-*`. Each subject folder should contain masked FLAIR, T1, T2, and lacune annotation files following the naming pattern used in the original notebooks, for example:

```text
Task3/
└── sub-101/
    ├── sub-101_space-T1_desc-masked_FLAIR.nii.gz
    ├── sub-101_space-T1_desc-masked_T1.nii.gz
    ├── sub-101_space-T1_desc-masked_T2.nii.gz
    └── sub-101_space-T1_desc-Rater2_Lacunes.nii.gz
```

Configure local paths with environment variables:

```bash
export LACUNE_DATA_ROOT=/path/to/Task3
export LACUNE_PROJECT_PATH=/path/to/project/root
```

`LACUNE_PROJECT_PATH` is used as the root for generated checkpoints, caches, and model-weight folders.

## Quick CLI Usage

Use `inference/predict_subject.py` to run a trained checkpoint on one subject:

```bash
python inference/predict_subject.py \
  --flair /path/to/sub-001_space-T1_desc-masked_FLAIR.nii.gz \
  --t1 /path/to/sub-001_space-T1_desc-masked_T1.nii.gz \
  --t2 /path/to/sub-001_space-T1_desc-masked_T2.nii.gz \
  --checkpoint "Model Weight/A4b Deep Supervision (DM)/fold1_best.pth" \
  --output outputs/sub-001_lacune_mask.nii.gz
```

Optional arguments:


The CLI automatically handles standard checkpoints and deep-supervision checkpoints. Inputs are reoriented to RAS, resampled to 1 mm isotropic spacing, intensity-normalized, and processed with sliding-window inference. The output mask is saved as a NIfTI file in the preprocessed RAS 1 mm space.

## Training

Run one experiment script at a time:

```bash
python training/train_candidate_generator_cv.py
python training/train_distance_map_cv.py
python training/train_deep_supervision_cv.py
```

Each script preserves the original 5-fold cross-validation workflow and writes checkpoints under `LACUNE_PROJECT_PATH/Model Weights/`.

## Experiment Evaluation

Evaluate the released Experiment A variants:

```bash
python inference/evaluate_experiment_a.py
```

The evaluation script expects fold checkpoints named like `fold1_best.pth`, `fold2_best.pth`, etc. under the variant directories configured in the script.


The visualization code keeps the original interactive viewer logic, so it is best run in an IPython/Jupyter-compatible environment.
