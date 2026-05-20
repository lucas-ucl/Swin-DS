#!/usr/bin/env python
"""Baseline Swin-UNETR training with 5-fold cross-validation.

A standard Swin-UNETR segmentation model without distance-map guidance
or deep supervision. Serves as the baseline for comparison
with the proposed Swin-DS variants.

Configure paths with environment variables:
    LACUNE_DATA_ROOT=/path/to/Task3
    LACUNE_PROJECT_PATH=/path/to/project_or_weights_root
"""

import copy
import glob
import os
import random
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    Spacingd,
)
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    message=".*Num foregrounds 0.*unable to generate class balanced samples.*",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_KEYS = ["flair", "t1", "t2"]
ALL_KEYS = IMG_KEYS + ["mask"]


def get_config():
    """Return a dict of all training hyperparameters and paths."""
    data_root = os.environ.get("LACUNE_DATA_ROOT", "./data/Task3")
    project_path = Path(os.environ.get("LACUNE_PROJECT_PATH", "."))
    weight_path = project_path / "Model Weights"
    cv_weight_path = weight_path / "baseline CV"
    cv_weight_path.mkdir(parents=True, exist_ok=True)

    return {
        "data_root": data_root,
        "project_path": project_path,
        "weight_path": weight_path,
        "cv_weight_path": cv_weight_path,
        "pretrained_file": weight_path / "model_swinvit.pt",
        "num_folds": 5,
        "max_folds": int(os.environ.get("LACUNE_MAX_FOLDS", "5")),
        "max_train_batches": int(os.environ.get("LACUNE_MAX_TRAIN_BATCHES", "0")),
        "max_val_batches": int(os.environ.get("LACUNE_MAX_VAL_BATCHES", "0")),
        "allow_random_init": os.environ.get("LACUNE_ALLOW_RANDOM_INIT", "0") == "1",
        "max_epochs": int(os.environ.get("LACUNE_MAX_EPOCHS", "200")),
        "early_stop_patience": int(os.environ.get("LACUNE_EARLY_STOP_PATIENCE", "200")),
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "seed": 24,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def load_subjects(data_root: str, has_lacunes_only: bool = False):
    """Scan data_root for subject folders and return a list of dicts."""
    subfolders = sorted([f for f in os.listdir(data_root) if f.startswith("sub-")])
    subjects = []

    for subf in subfolders:
        sub_dir = os.path.join(data_root, subf)
        if not os.path.isdir(sub_dir):
            continue

        sub_id = int(subf.split("-")[-1])
        rater_id = 2 if 101 <= sub_id <= 106 else 4

        flair = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_FLAIR.nii*"))
        t1 = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_T1.nii*"))
        t2 = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_T2.nii*"))
        mask = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-Rater{rater_id}_Lacunes.nii*"))

        if len(flair) != 1 or len(t1) != 1 or len(t2) != 1 or len(mask) != 1:
            continue

        mask_data = nib.load(mask[0]).get_fdata()
        subject_has_lacunes = bool(np.any(mask_data > 0))

        if has_lacunes_only and not subject_has_lacunes:
            continue

        subjects.append({
            "subject_id": subf,
            "flair": flair[0],
            "t1": t1[0],
            "t2": t2[0],
            "mask": mask[0],
            "has_lacunes": subject_has_lacunes,
        })

    print(f"Total subjects loaded: {len(subjects)}")
    return subjects


def create_fold_splits(subjects, num_folds: int, seed: int):
    """Create stratified K-fold splits based on lacune presence."""
    labels = np.array([int(s["has_lacunes"]) for s in subjects])
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_splits = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        train_files = [subjects[i] for i in tr_idx]
        val_files = [subjects[i] for i in val_idx]
        fold_splits.append({"train": train_files, "val": val_files})

        val_ids = [s["subject_id"] for s in val_files]
        print(f"Fold {fold_idx} validation ({len(val_ids)}): {', '.join(val_ids)}")

    return fold_splits


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_train_transforms():
    """Build training data augmentation pipeline."""
    return Compose([
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Orientationd(keys=ALL_KEYS, axcodes="RAS"),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=(1.0, 1.0, 1.0),
            mode=("trilinear", "trilinear", "trilinear", "nearest"),
            align_corners=True,
        ),
        NormalizeIntensityd(keys=IMG_KEYS),
        CropForegroundd(keys=ALL_KEYS, source_key="flair", margin=10, allow_smaller=True),
        RandCropByPosNegLabeld(
            keys=ALL_KEYS, label_key="mask",
            spatial_size=(96, 96, 96), pos=3, neg=1,
            num_samples=4, image_key="flair",
        ),
        RandFlipd(keys=ALL_KEYS, prob=0.25, spatial_axis=[0]),
        RandFlipd(keys=ALL_KEYS, prob=0.25, spatial_axis=[1]),
        RandFlipd(keys=ALL_KEYS, prob=0.25, spatial_axis=[2]),
        RandAdjustContrastd(keys=IMG_KEYS, prob=0.5, gamma=(0.5, 1.5)),
        EnsureTyped(keys=ALL_KEYS),
    ])


def build_val_transforms():
    """Build validation data pipeline (no augmentation)."""
    return Compose([
        LoadImaged(keys=ALL_KEYS),
        EnsureChannelFirstd(keys=ALL_KEYS),
        Orientationd(keys=ALL_KEYS, axcodes="RAS"),
        Spacingd(
            keys=ALL_KEYS,
            pixdim=(1.0, 1.0, 1.0),
            mode=("trilinear", "trilinear", "trilinear", "nearest"),
            align_corners=True,
        ),
        NormalizeIntensityd(keys=IMG_KEYS),
        CropForegroundd(keys=ALL_KEYS, source_key="flair", margin=10, allow_smaller=True),
        EnsureTyped(keys=ALL_KEYS),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def make_dataloaders(train_files, val_files, train_transforms, val_transforms):
    """Build MONAI CacheDatasets and DataLoaders for a single fold."""
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=10,
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=10,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=list_data_collate,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(device, pretrained_file: Path, allow_random_init: bool):
    """Create Swin-UNETR and load SSL-pretrained weights."""
    model = SwinUNETR(
        in_channels=3,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True,
    ).to(device)

    if pretrained_file.exists():
        model.load_from(weights=torch.load(pretrained_file, map_location=device))
        print(f"Loaded SSL weights from {pretrained_file}")
    elif allow_random_init:
        print(f"WARNING: {pretrained_file} not found; using random init.")
    else:
        raise FileNotFoundError(f"Missing SSL pretrained weights: {pretrained_file}")

    return model


# ---------------------------------------------------------------------------
# Training loop (single fold)
# ---------------------------------------------------------------------------


def train_one_fold(
    fold: int,
    train_loader,
    val_loader,
    device,
    cfg: dict,
):
    """Train and validate a single fold. Returns the best validation Dice."""
    model = build_model(device, cfg["pretrained_file"], cfg["allow_random_init"])

    loss_fn = DiceFocalLoss(
        include_background=False, to_onehot_y=True, softmax=True,
        lambda_dice=1.0, lambda_focal=1.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=50,
        threshold=1e-4, verbose=True, min_lr=1e-6,
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_softmax = nn.Softmax(dim=1)
    scaler = GradScaler()

    best_dice, best_state, epochs_no_improve = 0.0, None, 0
    max_train = cfg["max_train_batches"]
    max_val = cfg["max_val_batches"]

    for epoch in range(1, cfg["max_epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{cfg['max_epochs']}] — LR {lr:.2e}")

        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        dice_metric.reset()

        for batch_idx, batch in enumerate(train_loader, 1):
            if max_train and batch_idx > max_train:
                break
            inputs = torch.cat([batch[k].to(device) for k in IMG_KEYS], dim=1)
            targets = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=True):
                logits = model(inputs)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(post_softmax(logits), 1, keepdim=True)
                dice_metric(preds, (targets > 0.5))

        n_train = min(len(train_loader), max_train) if max_train else len(train_loader)
        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"   Train | Loss {epoch_loss / n_train:.4f}  Dice {train_dice:.4f}")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        dice_metric.reset()

        with torch.no_grad():
            for vbatch_idx, vbatch in enumerate(val_loader, 1):
                if max_val and vbatch_idx > max_val:
                    break
                vinputs = torch.cat([vbatch[k].to(device) for k in IMG_KEYS], dim=1)
                vtargets = vbatch["mask"].to(device)

                vlogits = sliding_window_inference(
                    vinputs, roi_size=(128, 128, 128),
                    sw_batch_size=1, overlap=0.6, predictor=model,
                )
                vloss = loss_fn(vlogits, vtargets)
                val_loss += vloss.item()

                vpreds = torch.argmax(post_softmax(vlogits), 1, keepdim=True)
                dice_metric(vpreds, (vtargets > 0.5))

        n_val = min(len(val_loader), max_val) if max_val else len(val_loader)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        scheduler.step(val_dice)
        print(f"   Val   | Loss {val_loss / n_val:.4f}  Dice {val_dice:.4f}")

        # ---- Checkpoint ----
        if val_dice > best_dice + 1e-4:
            best_dice, epochs_no_improve = val_dice, 0
            best_state = copy.deepcopy(model.state_dict())
            ckpt_path = cfg["cv_weight_path"] / f"fold{fold}_best.pth"
            torch.save(best_state, ckpt_path)
            print(f"   >> New best Dice! Saved to {ckpt_path.name}")
        else:
            epochs_no_improve += 1
            print(f"   -- No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= cfg["early_stop_patience"]:
            print("\n  Early stopping triggered.")
            break

    return best_dice


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(cfg["seed"])

    # Load data
    subjects = load_subjects(cfg["data_root"])
    fold_splits = create_fold_splits(subjects, cfg["num_folds"], cfg["seed"])

    # Build transforms
    train_transforms = build_train_transforms()
    val_transforms = build_val_transforms()

    # Cross-validation loop
    active_splits = fold_splits[: cfg["max_folds"]]
    fold_results = []

    for fold, split in enumerate(active_splits, 1):
        print(f"\n\n{'=' * 20}  FOLD {fold}/{len(active_splits)}  {'=' * 20}\n")

        train_loader, val_loader = make_dataloaders(
            split["train"], split["val"], train_transforms, val_transforms,
        )
        best_dice = train_one_fold(fold, train_loader, val_loader, device, cfg)
        fold_results.append(best_dice)
        print(f"\nFold {fold} complete. Best Val Dice = {best_dice:.4f}")
        torch.cuda.empty_cache()

    # Summary
    print("\n\n========== CROSS-VALIDATION SUMMARY ==========")
    for i, d in enumerate(fold_results, 1):
        print(f"  Fold {i}: best Dice = {d:.4f}")
    print(f"  Mean best Dice = {np.mean(fold_results):.4f}")


if __name__ == "__main__":
    main()
