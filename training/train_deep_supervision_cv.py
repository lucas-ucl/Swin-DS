#!/usr/bin/env python
"""Converted from the deep-supervision cross-validation notebook.

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
CV_WEIGHT_PATH = WEIGHT_PATH / "deep supervision CV 2"
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
# ================================================================
# Module 5 ─ Model Definition & Weights   ★ NEW DEEP SUPERVISION
# ================================================================
class SwinUNETR_DS(nn.Module):
    """
    Swin‑UNETR with deep supervision (4 scales: 1/8, 1/4, 1/2, 1).
    Each output has 3 channels: 2 for segmentation, 1 for distance regression.
    """
    def __init__(self, in_channels=3, out_channels=3, feature_size=48,
                 scales=(0.125, 0.25, 0.5, 1.0)):
        super().__init__()
        self.base = SwinUNETR(
            in_channels=in_channels, out_channels=out_channels,
            feature_size=feature_size, use_checkpoint=True, use_v2=True
        )
        self.scales = scales
    def load_from(self, weights):      # for SSL weight load
        self.base.load_from(weights)
    def forward(self, x):
        fine = self.base(x)
        outs = []
        for s in self.scales:
            outs.append(fine if s == 1.0 else
                         F.interpolate(fine, scale_factor=s, mode="trilinear",
                                       align_corners=False, recompute_scale_factor=True))
        return outs     # coarse → fine

# %% cell 11
# ================================================================
# Module 5 ─ Loss Classes & Helper Metrics   ★ UNCHANGED LOGIC
# ================================================================
class BoundaryWeightedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def _match_size(self, src, ref, mode):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode=mode, align_corners=False)
        return src
    def forward(self, logits, mask, dist_map):
        probs   = torch.softmax(logits, dim=1)
        prob_fg = probs[:, 1:2, ...]
        mask     = self._match_size(mask.float(), prob_fg, mode="nearest")
        dist_map = self._match_size(dist_map,       prob_fg, mode="trilinear")
        w = dist_map
        dims = tuple(range(2, prob_fg.ndim))
        inter = (w * prob_fg * mask).sum(dims)
        fp    = (w * prob_fg * (1 - mask)).sum(dims)
        fn    = (w * (1 - prob_fg) * mask).sum(dims)
        tversky = (inter + self.smooth) / (inter + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()

class FocalMSELoss(nn.Module):
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

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))
    def forward(self, loss_seg, loss_reg):
        prec_seg = torch.exp(-self.log_vars[0])
        prec_reg = torch.exp(-self.log_vars[1])
        return (prec_seg * loss_seg + self.log_vars[0] +
                prec_reg * loss_reg + self.log_vars[1])

# helper metric
def rmse(pred, tgt):
    return torch.sqrt(nn.functional.mse_loss(pred, tgt))

# %% [markdown] cell 12
# ## 6. Main Loop

# %% cell 13
# ================================================================
# Module 6 ─ 5‑Fold CV Training Loop   ★ REWRITTEN
# ================================================================
# ----- CV training -----
max_epochs          = int(os.environ.get("LACUNE_MAX_EPOCHS", "200"))
early_stop_patience = int(os.environ.get("LACUNE_EARLY_STOP_PATIENCE", str(max_epochs)))
post_softmax        = nn.Softmax(dim=1)
ds_decay            = 0.5
ds_weights          = [ds_decay ** (3 - i) for i in range(4)]   # 0.125,0.25,0.5,1

fold_best_dice = []
active_fold_splits = fold_splits[:MAX_FOLDS]

for fold, split in enumerate(active_fold_splits, 1):
    print(f"\n\n================  FOLD {fold}/{len(active_fold_splits)}  ================\n")
    train_loader, val_loader = make_dataloaders(split["train"], split["val"])

    # -------- model & optimiser --------
    model = SwinUNETR_DS().to(device)
    load_ssl_pretrained(model, WEIGHT_PATH / "model_swinvit.pt")

    loss_seg  = BoundaryWeightedTverskyLoss()
    loss_reg  = FocalMSELoss()
    comb_loss = UncertaintyLoss().to(device)

    optimiser = optim.AdamW(
        list(model.parameters()) + list(comb_loss.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=50,
        threshold=1e-4, verbose=True, min_lr=1e-6
    )
    dice_tr = DiceMetric(include_background=False, reduction="mean")
    dice_va = DiceMetric(include_background=False, reduction="mean")

    scaler = GradScaler()
    best_dice, best_state, epochs_no_improve = 0., None, 0

    # ------------------ epoch loop ------------------
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch [{epoch}/{max_epochs}] — LR {optimiser.param_groups[0]['lr']:.2e}")

        # ---------- TRAIN ----------
        model.train(); ep_loss = 0.; dice_tr.reset()
        for batch_idx, batch in enumerate(train_loader, 1):
            if MAX_TRAIN_BATCHES and batch_idx > MAX_TRAIN_BATCHES:
                break
            inp  = torch.cat([batch[k].to(device) for k in IMG_KEYS], 1)
            msk  = batch["mask"].to(device)
            dst  = batch["dist"].to(device)

            optimiser.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=True):
                outs = model(inp)        # 4 scales
                seg_acc, reg_acc = 0., 0.
                for w, o in zip(ds_weights, outs):
                    seg_log = o[:, 0:2]; dist_p = torch.sigmoid(o[:, 2:3])
                    m = F.interpolate(msk.float(), size=dist_p.shape[2:], mode="nearest")
                    d = F.interpolate(dst,        size=dist_p.shape[2:], mode="trilinear", align_corners=False)
                    seg_acc += w * loss_seg(seg_log, m, d)
                    reg_acc += w * loss_reg(dist_p, d)
                loss = comb_loss(seg_acc, reg_acc)

            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update()
            ep_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(post_softmax(outs[-1][:, 0:2]), 1, keepdim=True)
                dice_tr(preds, (msk > 0.5))

        n_train_iters = min(len(train_loader), MAX_TRAIN_BATCHES) if MAX_TRAIN_BATCHES else len(train_loader)
        ep_loss /= n_train_iters
        tr_dice   = dice_tr.aggregate().item(); dice_tr.reset()
        print(f"   Train ‖ Loss {ep_loss:.4f}  Dice {tr_dice:.4f}")

        # ---------- VALIDATE ----------
        model.eval(); v_loss = 0.; rmse_val = 0.; dice_va.reset()
        with torch.no_grad():
            for val_batch_idx, vb in enumerate(val_loader, 1):
                if MAX_VAL_BATCHES and val_batch_idx > MAX_VAL_BATCHES:
                    break
                vi = torch.cat([vb[k].to(device) for k in IMG_KEYS], 1)
                vm = vb["mask"].to(device)
                vd = vb["dist"].to(device)

                vout = sliding_window_inference(
                    vi, roi_size=(128,128,128), sw_batch_size=1, overlap=0.6,
                    predictor=lambda x: model(x)[-1]   # only finest scale
                )
                v_seg = vout[:, 0:2]; v_dist = torch.sigmoid(vout[:, 2:3])
                vm_r  = F.interpolate(vm.float(), size=v_dist.shape[2:], mode="nearest")
                vd_r  = F.interpolate(vd, size=v_dist.shape[2:], mode="trilinear", align_corners=False)

                vs = loss_seg(v_seg, vm_r, vd_r)
                vr = loss_reg(v_dist, vd_r)
                l  = comb_loss(vs, vr)
                v_loss += l.item()

                preds = torch.argmax(post_softmax(v_seg), 1, keepdim=True)
                dice_va(preds, (vm_r > 0.5))
                rmse_val += rmse(v_dist, vd_r).item()

        n_val_iters = min(len(val_loader), MAX_VAL_BATCHES) if MAX_VAL_BATCHES else len(val_loader)
        v_loss   /= n_val_iters
        v_dice    = dice_va.aggregate().item(); dice_va.reset()
        rmse_val /= n_val_iters
        scheduler.step(v_dice)

        print(f"   Val   ‖ Loss {v_loss:.4f}  Dice {v_dice:.4f}  RMSE {rmse_val:.4f}")

        # ---------- CHECKPOINT ----------
        if v_dice > best_dice + 1e-4:
            best_dice, epochs_no_improve = v_dice, 0
            best_state = copy.deepcopy(model.state_dict())
            ckpt = CV_WEIGHT_PATH / f"fold{fold}_best.pth"
            torch.save(best_state, ckpt)
            print(f"   ✓  New best Dice! ➜ {ckpt.name}")
        else:
            epochs_no_improve += 1
            print(f"   •  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print("\n  Early stopping — Dice plateau."); break

    fold_best_dice.append(best_dice)
    print(f"\nFold {fold} complete. Best Val Dice = {best_dice:.4f}")
    torch.cuda.empty_cache()

# -------------- summary --------------
print("\n\n==========  CROSS‑VALIDATION SUMMARY  ==========")
for i, d in enumerate(fold_best_dice, 1):
    print(f"Fold {i}: best Dice = {d:.4f}")
print(f"Mean best Dice across folds = {np.mean(fold_best_dice):.4f}")
