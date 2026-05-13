#!/usr/bin/env python
"""Converted from Training/1-2. Candidate Generator.ipynb.

This script preserves the original notebook execution order while removing
Colab-only installation and Google Drive mount commands. Configure paths with:
  LACUNE_DATA_ROOT=/path/to/Task3
  LACUNE_PROJECT_PATH=/path/to/project_or_weights_root
"""

# %% cell 1
## Module 0: Import Dependence


# -----------------------
# Standard library imports
# -----------------------
import glob
import os
import random
import warnings
import copy

# -----------------------
# Third-party imports
# -----------------------
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt as dte
from scipy.ndimage import label
from tqdm import tqdm
from pathlib import Path
import cc3d
from sklearn.model_selection import StratifiedKFold

# -----------------------
# PyTorch imports
# -----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

# -----------------------
# MONAI imports
# -----------------------
import monai
from monai.transforms import (
    MapTransform,
    Transform,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    CropForegroundd,
    EnsureTyped,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    ResizeD,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    CopyItemsd
)
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, TverskyLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.config.type_definitions import KeysCollection
from monai.utils import set_determinism

# Suppress the specific warning about "Num foregrounds 0..."
warnings.filterwarnings(
    "ignore",
    message=".*Num foregrounds 0.*unable to generate class balanced samples, setting `pos_ratio` to 0.*"
)

# %% cell 2
## Module 1: Configuration


# -----------------------------------------------------------------
# ▸ Core paths
# -----------------------------------------------------------------
DATA_ROOT = os.environ.get("LACUNE_DATA_ROOT", "./data/Task3")
PROJECT_PATH = Path(os.environ.get("LACUNE_PROJECT_PATH", "."))
WEIGHT_PATH    = PROJECT_PATH / "Model Weights"
CV_WEIGHT_PATH = WEIGHT_PATH / "distance map CV 3"
CV_WEIGHT_PATH.mkdir(parents=True, exist_ok=True)

NUM_FOLDS = 5
MAX_FOLDS = int(os.environ.get("LACUNE_MAX_FOLDS", str(NUM_FOLDS)))
MAX_TRAIN_BATCHES = int(os.environ.get("LACUNE_MAX_TRAIN_BATCHES", "0"))
MAX_VAL_BATCHES = int(os.environ.get("LACUNE_MAX_VAL_BATCHES", "0"))
ALLOW_RANDOM_INIT = os.environ.get("LACUNE_ALLOW_RANDOM_INIT", "0") == "1"
has_lacunes_only = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# -----------------------------------------------------------------
# ▸ Reproducibility
# -----------------------------------------------------------------
seed = 24
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_determinism(seed=seed)


# -------------------------
# Load subjects
# -------------------------

subfolders = sorted([f for f in os.listdir(DATA_ROOT) if f.startswith("sub-")])
all_subjects = []

for subf in subfolders:
    sub_dir = os.path.join(DATA_ROOT, subf)
    if not os.path.isdir(sub_dir):
        continue

    # Assign Raters
    sub_id = int(subf.split("-")[-1])
    if 101 <= sub_id <= 106:
        rater_id = 2
    else:
        rater_id = 4

    # Locate relevant files
    flair_file = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_FLAIR.nii*"))
    t1_file    = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_T1.nii*"))
    t2_file    = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-masked_T2.nii*"))
    mask_file  = glob.glob(os.path.join(sub_dir, f"{subf}_space-T1_desc-Rater{rater_id}_Lacunes.nii*"))

    # Skip if any file is missing
    if len(flair_file) != 1 or len(t1_file) != 1 or len(t2_file) != 1 or len(mask_file) != 1:
        continue

    # Check if the mask has lacunes by reading non-zero voxels
    mask_data = nib.load(mask_file[0]).get_fdata()
    subject_has_lacunes = bool(np.any(mask_data > 0))

    # Optionally skip subjects who don't have lacunes
    if has_lacunes_only and not subject_has_lacunes:
        continue

    all_subjects.append({
        "subject_id": subf,
        "flair": flair_file[0],
        "t1":    t1_file[0],
        "t2":    t2_file[0],
        "mask":  mask_file[0],
        "has_lacunes": subject_has_lacunes
    })

print(f"Total subjects loaded: {len(all_subjects)}")

# %% [markdown] cell 3
# ## 2. Create Train / Valid Sets

# %% cell 4
# Module 2: Create Train & Valid Sets

# ---------- 5‑fold stratified splits ----------
labels = np.array([int(s["has_lacunes"]) for s in all_subjects])
skf    = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

fold_splits = []
for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
    train_files = [all_subjects[i] for i in tr_idx]
    val_files   = [all_subjects[i] for i in val_idx]
    fold_splits.append({"train": train_files, "val": val_files})

    # ---- Print (instead of save) validation subject IDs ----
    val_ids = [s["subject_id"] for s in val_files]
    print(f"\nFold {fold_idx} validation IDs ({len(val_ids)}):")
    print(", ".join(val_ids))

print(f"\n{NUM_FOLDS}-fold splits created. Validation IDs printed above.")


def load_ssl_pretrained(model, pretrained_file: Path) -> None:
    if pretrained_file.exists():
        model.load_from(weights=torch.load(pretrained_file, map_location=device))
        return
    if ALLOW_RANDOM_INIT:
        print(f"WARNING: {pretrained_file} not found; using random initialization for smoke testing.")
        return
    raise FileNotFoundError(f"Missing SSL pretrained weights: {pretrained_file}")

# %% [markdown] cell 5
# ## 3. Transforms Pipeline

# %% cell 6
# ================================================================
# Module 3a ─ Custom Transform (distance map)  ★ UNCHANGED
# ================================================================
class ComputeDistanceMapd(MapTransform):
    """
    Create inverse‑distance map (strictly in (0,1]) from a binary mask.
        dist_map = 1 / (1 + 0.1 * edt_distance)
    """
    def __init__(self, label_key: str = "mask", dist_key: str = "dist"):
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.dist_key  = dist_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.label_key][0]                        # shape (D,H,W)
        dist = dte(1 - mask)
        d[self.dist_key] = (1.0 / (1.0 + 0.1 * dist))[None, ...].astype(np.float32)
        return d

# %% cell 7
# ================================================================
# Module 3 ─ Transform Pipelines   ★ UNCHANGED LOGIC
# ================================================================
IMG_KEYS = ["flair", "t1", "t2"]
LAB_KEYS = ["mask", "dist"]
ALL_KEYS = IMG_KEYS + LAB_KEYS

train_transforms = Compose([
    LoadImaged(keys=ALL_KEYS[:-1]),              # «dist» generated later
    EnsureChannelFirstd(keys=ALL_KEYS[:-1]),
    Orientationd(keys=ALL_KEYS[:-1], axcodes="RAS"),
    Spacingd(
        keys=ALL_KEYS[:-1],
        pixdim=(1.0, 1.0, 1.0),
        mode=("trilinear","trilinear","trilinear","nearest"),
        align_corners=True
    ),
    NormalizeIntensityd(keys=IMG_KEYS),
    CropForegroundd(keys=ALL_KEYS[:-1], source_key="flair", margin=10, allow_smaller=True),
    ComputeDistanceMapd(label_key="mask", dist_key="dist"),
    RandCropByPosNegLabeld(
        keys=ALL_KEYS[:-1], label_key="mask",
        spatial_size=(96,96,96), pos=3, neg=1,
        num_samples=4, image_key="flair"
    ),
    RandFlipd(keys=ALL_KEYS[:-1], prob=0.25, spatial_axis=[0]),
    RandFlipd(keys=ALL_KEYS[:-1], prob=0.25, spatial_axis=[1]),
    RandFlipd(keys=ALL_KEYS[:-1], prob=0.25, spatial_axis=[2]),
    RandAdjustContrastd(keys=IMG_KEYS, prob=0.5, gamma=(0.5, 1.5)),
    EnsureTyped(keys=ALL_KEYS),
])

val_transforms = Compose([
    LoadImaged(keys=ALL_KEYS[:-1]),
    EnsureChannelFirstd(keys=ALL_KEYS[:-1]),
    Orientationd(keys=ALL_KEYS[:-1], axcodes="RAS"),
    Spacingd(
        keys=ALL_KEYS[:-1],
        pixdim=(1.0, 1.0, 1.0),
        mode=("trilinear","trilinear","trilinear","nearest"),
        align_corners=True
    ),
    NormalizeIntensityd(keys=IMG_KEYS),
    CropForegroundd(keys=ALL_KEYS[:-1], source_key="flair", margin=10, allow_smaller=True),
    ComputeDistanceMapd(label_key="mask", dist_key="dist"),
    EnsureTyped(keys=ALL_KEYS),
])

# %% [markdown] cell 8
# ## 4. Datasets & DataLoaders

# %% cell 9
# Module 4: Dataset & DataLoaders

def make_dataloaders(train_files, val_files):
    """
    Build MONAI CacheDatasets + DataLoaders for a single fold.
    """
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=10
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=10
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=list_data_collate
    )

    return train_loader, val_loader

# %% cell 10
class BoundaryWeightedTverskyLoss(nn.Module):
    """
    Tversky loss weighted by the precomputed distance map.
    The implementation aligns mask and distance-map sizes to the prediction.
    """
    def __init__(self, alpha=0.1, beta=0.9, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def _match_size(self, src, ref, mode):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode=mode, align_corners=False)
        return src

    def forward(self, logits, mask, dist_map):
        # logits: (B,2,...) -> softmax foreground channel.
        probs   = torch.softmax(logits, dim=1)
        prob_fg = probs[:, 1:2, ...]
        # Match target sizes to the prediction.
        mask     = self._match_size(mask.float(), prob_fg, mode="nearest")
        dist_map = self._match_size(dist_map,       prob_fg, mode="trilinear")
        # Distance-map weighting.
        w = dist_map
        dims = tuple(range(2, prob_fg.ndim))
        inter = (w * prob_fg * mask).sum(dims)
        fp    = (w * prob_fg * (1 - mask)).sum(dims)
        fn    = (w * (1 - prob_fg) * mask).sum(dims)
        tversky = (inter + self.smooth) / (inter + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()

class FocalMSELoss(nn.Module):
    """
    Focal-MSE loss for distance-map regression.
    """
    def __init__(self, gamma=2.5, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma, self.reduction, self.eps = gamma, reduction, eps

    def forward(self, preds, targets):
        mse = (preds - targets) ** 2
        weights = (targets.abs() + self.eps) ** self.gamma
        loss = weights * mse
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        return loss.mean()


# helper metric
def rmse(pred, tgt):
    return torch.sqrt(nn.functional.mse_loss(pred, tgt))

# %% [markdown] cell 11
# ## 6. Main Loop

# %% cell 12
# ===== Loss hyperparameters =====
TVERSKY_ALPHA  = 0.1
TVERSKY_BETA   = 0.9
TVERSKY_SMOOTH = 1e-6
FOCAL_GAMMA    = 2.5   # FocalMSELoss gamma.

max_epochs          = int(os.environ.get("LACUNE_MAX_EPOCHS", "200"))
early_stop_patience = int(os.environ.get("LACUNE_EARLY_STOP_PATIENCE", str(max_epochs)))
post_softmax        = nn.Softmax(dim=1)

fold_best_dice = []
active_fold_splits = fold_splits[:MAX_FOLDS]

for fold, split in enumerate(active_fold_splits, 1):
    print(f"\n\n================  FOLD {fold}/{len(active_fold_splits)}  ================\n")

    # 1) loaders
    train_loader, val_loader = make_dataloaders(split["train"], split["val"])

    # 2) model: 3 output channels, seg(2) + dist(1)
    model = SwinUNETR(
        in_channels=3,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
        use_v2=True
    ).to(device)
    load_ssl_pretrained(model, WEIGHT_PATH / "model_swinvit.pt")

    # 3) losses & optimizer; total loss = seg + reg.
    loss_fn_seg = BoundaryWeightedTverskyLoss(alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=TVERSKY_SMOOTH)
    loss_fn_reg = FocalMSELoss(gamma=FOCAL_GAMMA, reduction='mean')

    optimiser = optim.AdamW(
        model.parameters(),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=50,
        threshold=1e-4, verbose=True, min_lr=1e-6
    )
    dice_metric_train = DiceMetric(include_background=False, reduction="mean")
    dice_metric_val   = DiceMetric(include_background=False, reduction="mean")

    scaler = GradScaler()
    best_dice, best_state, epochs_no_improve = 0.0, None, 0

    # ------------------ EPOCH LOOP ------------------
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch [{epoch}/{max_epochs}] — LR {optimiser.param_groups[0]['lr']:.2e}")

        # ------------ TRAIN ------------
        model.train(); epoch_loss = 0.0; epoch_loss_seg=0.0; epoch_loss_reg=0.0
        dice_metric_train.reset()
        for batch_idx, batch in enumerate(train_loader, 1):
            if MAX_TRAIN_BATCHES and batch_idx > MAX_TRAIN_BATCHES:
                break
            inputs   = torch.cat([batch[k].to(device) for k in IMG_KEYS], dim=1)  # (B,3,...)
            tgt_mask = batch["mask"].to(device)                                   # (B,1,...)
            tgt_dist = batch["dist"].to(device)                                   # (B,1,...)

            optimiser.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=True):
                logits      = model(inputs)                 # (B,3,…)
                seg_logits  = logits[:, 0:2, ...]
                dist_pred   = torch.sigmoid(logits[:, 2:3, ...])

                # Align the regression target to the prediction size.
                if tgt_dist.shape[2:] != dist_pred.shape[2:]:
                    tgt_dist_res = torch.nn.functional.interpolate(
                        tgt_dist, size=dist_pred.shape[2:], mode="trilinear", align_corners=False
                    )
                else:
                    tgt_dist_res = tgt_dist

                # Segmentation term; the loss aligns mask/dist internally.
                loss_seg = loss_fn_seg(seg_logits, tgt_mask, tgt_dist_res)
                # Regression term.
                loss_reg = loss_fn_reg(dist_pred, tgt_dist_res)
                # Equal weighting.
                loss = loss_seg + loss_reg

            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update()

            epoch_loss     += loss.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_reg += loss_reg.item()

            with torch.no_grad():
                preds = torch.argmax(post_softmax(seg_logits), 1, keepdim=True)
                dice_metric_train(preds, (tgt_mask > 0.5))

        n_train_iters = min(len(train_loader), MAX_TRAIN_BATCHES) if MAX_TRAIN_BATCHES else len(train_loader)
        epoch_loss     /= n_train_iters
        epoch_loss_seg /= n_train_iters
        epoch_loss_reg /= n_train_iters
        train_dice      = dice_metric_train.aggregate().item()
        dice_metric_train.reset()
        print(f"   Train ‖ Total {epoch_loss:.4f}  Seg {epoch_loss_seg:.4f}  Reg {epoch_loss_reg:.4f}  Dice {train_dice:.4f}")

        # ------------ VALIDATE ------------
        model.eval(); val_loss = 0.0; val_loss_seg=0.0; val_loss_reg=0.0; reg_rmse_val = 0.0
        dice_metric_val.reset()
        with torch.no_grad():
            for val_batch_idx, vbatch in enumerate(val_loader, 1):
                if MAX_VAL_BATCHES and val_batch_idx > MAX_VAL_BATCHES:
                    break
                vinputs = torch.cat([vbatch[k].to(device) for k in IMG_KEYS], dim=1)
                v_mask  = vbatch["mask"].to(device)
                v_dist  = vbatch["dist"].to(device)

                vout = sliding_window_inference(
                    vinputs, roi_size=(128,128,128),
                    sw_batch_size=1, overlap=0.6, predictor=model
                )
                v_seg_logits = vout[:, 0:2, ...]
                v_dist_pred  = torch.sigmoid(vout[:, 2:3, ...])

                # Align the regression target.
                if v_dist.shape[2:] != v_dist_pred.shape[2:]:
                    v_dist_res = torch.nn.functional.interpolate(
                        v_dist, size=v_dist_pred.shape[2:], mode="trilinear", align_corners=False
                    )
                else:
                    v_dist_res = v_dist

                vloss_seg = loss_fn_seg(v_seg_logits, v_mask, v_dist_res)
                vloss_reg = loss_fn_reg(v_dist_pred, v_dist_res)
                vloss     = vloss_seg + vloss_reg

                val_loss     += vloss.item()
                val_loss_seg += vloss_seg.item()
                val_loss_reg += vloss_reg.item()

                vpreds = torch.argmax(post_softmax(v_seg_logits), 1, keepdim=True)
                dice_metric_val(vpreds, (v_mask > 0.5))
                reg_rmse_val += rmse(v_dist_pred, v_dist_res).item()

        n_val_iters   = min(len(val_loader), MAX_VAL_BATCHES) if MAX_VAL_BATCHES else len(val_loader)
        val_loss     /= n_val_iters
        val_loss_seg /= n_val_iters
        val_loss_reg /= n_val_iters
        val_dice      = dice_metric_val.aggregate().item()
        reg_rmse_val /= n_val_iters
        dice_metric_val.reset()
        scheduler.step(val_dice)

        print(f"   Val   ‖ Total {val_loss:.4f}  Seg {val_loss_seg:.4f}  Reg {val_loss_reg:.4f}  Dice {val_dice:.4f}  RMSE {reg_rmse_val:.4f}")

        # ------------ CHECKPOINT ------------
        if val_dice > best_dice + 1e-4:
            best_dice, epochs_no_improve = val_dice, 0
            best_state = copy.deepcopy(model.state_dict())
            ckpt_path = CV_WEIGHT_PATH / f"fold{fold}_best.pth"
            torch.save(best_state, ckpt_path)
            print(f"   ✓  New best Dice! ➜ weights saved to {ckpt_path.name}")
        else:
            epochs_no_improve += 1
            print(f"   •  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print("\n  Early stopping — Dice plateau.")
            break

    # ---------- end‑of‑fold ----------
    fold_best_dice.append(best_dice)
    print(f"\nFold {fold} complete. Best Val Dice = {best_dice:.4f}")
    torch.cuda.empty_cache()

# ----------------– Summary –----------------
print("\n\n==========  CROSS‑VALIDATION SUMMARY  ==========")
for idx, d in enumerate(fold_best_dice, 1):
    print(f"Fold {idx}: best Dice = {d:.4f}")
print(f"Mean best Dice across folds = {np.mean(fold_best_dice):.4f}")
