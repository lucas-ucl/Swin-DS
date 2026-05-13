#!/usr/bin/env python
"""Converted from Inference/Stage 1_Inference.ipynb.

This script preserves the original notebook execution order while removing
Colab-only installation and Google Drive mount commands. Configure paths with:
  LACUNE_DATA_ROOT=/path/to/Task3
  LACUNE_PROJECT_PATH=/path/to/project_or_weights_root
"""

# %% [markdown] cell 0
# ## 0. Install Environments

# %% cell 1
## Monai
## Detection

# %% cell 2
## Module 0: Import Dependence


# -----------------------
# Standard library imports
# -----------------------
import glob
import os
import random
import warnings
import copy
import math
import gc

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
import statistics as st
from scipy.ndimage import distance_transform_edt as dte
from scipy.ndimage import label
import scipy.ndimage as ndi
from scipy.stats import wilcoxon

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
import cc3d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from matplotlib import colors as mcolors
from collections import OrderedDict


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
from monai.data import CacheDataset, list_data_collate, PersistentDataset
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
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# enable autocast only on CUDA
AMP_ENABLED  = torch.cuda.is_available()

# %% [markdown] cell 3
# ## Module 1: Configuration

# %% cell 4
## Module 1
## Configurations


## Core paths
DATA_ROOT = os.environ.get("LACUNE_DATA_ROOT", "./data/Task3")
PROJECT_PATH = Path(os.environ.get("LACUNE_PROJECT_PATH", "."))
WEIGHT_PATH      = Path(os.environ.get("LACUNE_WEIGHT_PATH", str(PROJECT_PATH / "Model Weights/Experiment A")))
Cache_Path       = Path(os.environ.get("LACUNE_CACHE_PATH", str(WEIGHT_PATH / "Cache")))


## Model Addresses
VARIANT_DIRS = OrderedDict({
    "A0":  WEIGHT_PATH / "A0 baseline CV",
    "A1":  WEIGHT_PATH / "A2 DM Seg + Reg",
    "A2":  WEIGHT_PATH / "A3 DM Uncertainty Weighting",
    "A3":  WEIGHT_PATH / "A4a Deep Supervision (Seg + DM)",

    # Existing DS(DM) you added before (kept as-is)
    "A4":  WEIGHT_PATH / "A4b Deep Supervision (DM)",

    # NEW: DS on segmentation only (out_channels=2)
    "A4c": WEIGHT_PATH / "A4c Deep Supervision (Seg)",
})

VARIANT_CFG = {
    "A0":  {"out_channels": 2, "is_ds": False},  # baseline segmentation only
    "A1":  {"out_channels": 3, "is_ds": False},  # seg + distance
    "A2":  {"out_channels": 3, "is_ds": False},  # seg + distance + uncertainty weighting
    "A3":  {"out_channels": 3, "is_ds": True},   # deep supervision (Seg + DM)

    # A4b (DM) - 3 channels (2 seg + 1 dm)
    "A4":  {"out_channels": 3, "is_ds": True},

    # A4c (Seg only) - 2 channels
    "A4c": {"out_channels": 2, "is_ds": True},
}


## Reproducibility
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
set_determinism(seed=seed)


## Constants
NUM_FOLDS            = 5
MAX_VARIANTS         = int(os.environ.get("LACUNE_MAX_VARIANTS", str(len(VARIANT_DIRS))))
MAX_FOLDS            = int(os.environ.get("LACUNE_MAX_FOLDS", str(NUM_FOLDS)))
MAX_VAL_SUBJECTS     = int(os.environ.get("LACUNE_MAX_VAL_SUBJECTS", "0"))
MAX_VAL_BATCHES      = int(os.environ.get("LACUNE_MAX_VAL_BATCHES", "0"))
CONNECTIVITY         = 26
MIN_VOX_PRED         = 5
NSD_TOL_MM           = 1.5
BOOT_N               = int(os.environ.get("LACUNE_BOOT_N", "2000"))
RNG_SEED             = 24
ROI_SIZE             = (128, 128, 128)
SPACING              = (1.0, 1.0, 1.0)
SW_BATCH_SIZE        = int(os.environ.get("LACUNE_SW_BATCH_SIZE", "1"))
OVERLAP              = float(os.environ.get("LACUNE_OVERLAP", "0.6"))
post_softmax         = nn.Softmax(dim=1)
IMG_KEYS             = ["flair", "t1", "t2"]

# Pretrained Swin-ViT SSL weights
SSL_PRETRAINED_FILE  = WEIGHT_PATH / "model_swinvit.pt"

# %% [markdown] cell 5
# ## Module 2: Load Subjects + Split data

# %% cell 6
## Module 2
## Load Subjects

subfolders = sorted([f for f in os.listdir(DATA_ROOT) if f.startswith("sub-")])
subjects = []

for subf in subfolders:
    d = Path(DATA_ROOT) / subf

    # Assign Raters
    sid_num = int(subf.split("-")[-1])
    rater   = 2 if 101 <= sid_num <= 106 else 4

    # Locate Relvant Files
    flair = list(d.glob(f"{subf}_space-T1_desc-masked_FLAIR.nii*"))
    t1    = list(d.glob(f"{subf}_space-T1_desc-masked_T1.nii*"))
    t2    = list(d.glob(f"{subf}_space-T1_desc-masked_T2.nii*"))
    mask  = list(d.glob(f"{subf}_space-T1_desc-Rater{rater}_Lacunes.nii*"))
    if not (len(flair)==len(t1)==len(t2)==len(mask)==1): continue

    mdata = nib.load(mask[0]).get_fdata()
    has_lac = bool((mdata>0).any())
    subjects.append({
        "sid": subf,
        "flair": str(flair[0]),
        "t1":    str(t1[0]),
        "t2":    str(t2[0]),
        "mask":  str(mask[0]),
        "label": int(has_lac)
    })

print(f"Total subjects: {len(subjects)}")

# %% cell 7
## Module 2
## Split Data

def create_cv_folds(subjects: list[dict], num_folds: int = 5, seed: int = 24) -> list[dict]:

    labels = np.array([s["label"] for s in subjects])
    skf    = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    folds = []

    print("\n--- Cross‑Validation Folds ---")
    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        train_subjects = [subjects[i] for i in tr_idx]
        val_subjects   = [subjects[i] for i in val_idx]
        val_sids       = [s["sid"] for s in val_subjects]

        folds.append({
            "train_indices":      tr_idx,
            "validation_indices": val_idx,
            "train":              train_subjects,
            "val":                val_subjects,
            "validation_sids":    val_sids
        })

        # --- Console output for transparency -------------------------
        print(f"Fold {fold_id}:")
        print(f"  Validation count : {len(val_idx)}")
        print(f"  Validation SIDs  : {val_sids}")
        print("-" * 40)

    return folds

# -------------------------------------------------
# Execute the split (subjects list already exists)
# -------------------------------------------------
folds = create_cv_folds(subjects=subjects, num_folds=5, seed=seed)

# %% [markdown] cell 8
# ## Module 3: Transforms Pipeline

# %% cell 9
## Module 3
## Validation Data Transforms

KEYS = ["flair", "t1", "t2", "mask"]

val_transforms = Compose([
    LoadImaged(keys=KEYS),
    EnsureChannelFirstd(keys=KEYS),
    Orientationd(keys=KEYS, axcodes="RAS"),
    Spacingd(
        keys=KEYS,
        pixdim=(1.0, 1.0, 1.0),
        mode=("trilinear", "trilinear", "trilinear", "nearest"),
        align_corners=True
    ),
    NormalizeIntensityd(keys=["flair", "t1", "t2"]),
    CropForegroundd(keys=KEYS, source_key="flair", margin=10, allow_smaller=True),
    EnsureTyped(keys=KEYS),
])

# %% [markdown] cell 10
# ## Module 4: Create Datasets & DataLoaders

# %% cell 11
## Module 4
## Dataset & DataLoaders

def make_dataloaders(train_files, val_files):

    val_ds = PersistentDataset(
        data=val_files,
        transform=val_transforms,
        cache_dir=str(Cache_Path / "val_a")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=list_data_collate
    )

    return val_loader

# %% [markdown] cell 12
# ## Module 5: Evaluation Functions

# %% cell 13
## Module 5
## Evaluation Functions

# ------------------------------------------------------------
# Connected components & centroids
# ------------------------------------------------------------
def connected_components_3d(mask: np.ndarray, conn: int = CONNECTIVITY, min_vox: int = 1) -> np.ndarray:
    """
    3D connected-component labeling with optional size filtering.
    - Labels components using `cc3d.connected_components`.
    - Removes components with fewer than `min_vox` voxels.
    - Remaps labels to consecutive integers 1..K (0 = background).
    """
    lbl = cc3d.connected_components(mask.astype(np.uint8), connectivity=conn)

    # Filter small components (vectorized)
    if min_vox > 1 and lbl.max() > 0:
        labels, counts = np.unique(lbl, return_counts=True)  # includes 0
        small = labels[(counts < min_vox) & (labels != 0)]
        if small.size > 0:
            lbl[np.isin(lbl, small)] = 0

    # Remap to 1..K (avoid cc3d.remap for compatibility)
    if lbl.max() > 0:
        uniq = np.unique(lbl)
        uniq = uniq[uniq != 0]
        lut = np.zeros(int(uniq.max()) + 1, dtype=np.int32)
        lut[uniq] = np.arange(1, uniq.size + 1, dtype=np.int32)
        lbl = lut[lbl.astype(np.int32, copy=False)]
    else:
        lbl = lbl.astype(np.int32, copy=False)
    return lbl

def component_centroids(lbl: np.ndarray):
    """
    Returns (centroids, ids) for labeled mask `lbl` (0 = background).
    Faster and equivalent to argwhere/mean: uses `scipy.ndimage.center_of_mass`.
    """
    ids = np.setdiff1d(np.unique(lbl), 0)
    if ids.size == 0:
        return np.empty((0, 3), float), ids
    # center_of_mass expects an "intensity" array and label indices
    cents = [np.argwhere(lbl == cid).mean(0) for cid in ids]
    # cents = ndi.center_of_mass(np.ones_like(lbl, dtype=np.float32), labels=lbl, index=ids.tolist())
    return np.asarray(cents, dtype=float), ids

# ------------------------------------------------------------
# Lesion matching & counts (detection metrics)
# ------------------------------------------------------------
def match_tp_pairs(lbl_pred: np.ndarray, lbl_gt: np.ndarray, thresh: float = 5.0):
    """
    One-to-one nearest-centroid matching within a voxel-distance threshold.
    Each GT lesion can be matched at most once. Returns list of (pred_id, gt_id).
    """
    p_cent, p_ids = component_centroids(lbl_pred)
    g_cent, g_ids = component_centroids(lbl_gt)
    if p_ids.size == 0 or g_ids.size == 0:
        return []

    matched_gt = set()
    tp_pairs = []
    for i, pid in enumerate(p_ids):
        dists = np.linalg.norm(p_cent[i] - g_cent, axis=1) if g_cent.size else np.array([])
        if dists.size == 0:
            continue
        j = int(dists.argmin())
        if (dists[j] < thresh) and (g_ids[j] not in matched_gt):
            tp_pairs.append((pid, g_ids[j]))
            matched_gt.add(g_ids[j])
    return tp_pairs

def lesion_counts(lbl_pred: np.ndarray, lbl_gt: np.ndarray, thresh: float = 5.0):
    """
    Lesion-level TP/FP/FN at a fixed threshold (0.5 upstream).
    Matching rule: nearest-centroid, each GT matched at most once.
    Returns (tp, fp, fn).
    """
    p_cent, p_ids = component_centroids(lbl_pred)
    g_cent, g_ids = component_centroids(lbl_gt)

    matched_gt = set()
    tp = fp = 0
    for i, pid in enumerate(p_ids):
        if g_cent.size == 0:
            fp += 1
            continue
        dists = np.linalg.norm(p_cent[i] - g_cent, axis=1)
        j = int(dists.argmin())
        if dists[j] < thresh and g_ids[j] not in matched_gt:
            tp += 1
            matched_gt.add(g_ids[j])
        else:
            fp += 1
    fn = len(g_ids) - len(matched_gt)
    return tp, fp, fn

# ------------------------------------------------------------
# Safe aggregations & bootstrap CI
# ------------------------------------------------------------
def safe_mean(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if len(vals) else math.nan

def safe_median(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if len(vals) else math.nan

def bootstrap_ci_mean(values, n_boot=BOOT_N, seed=RNG_SEED, ci=0.95):
    """
    Nonparametric bootstrap CI for the mean.
    Returns (mean, low, high). NaNs are removed automatically.
    """
    vals = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if vals.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n = vals.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(vals[idx])
    mean = float(np.mean(vals))
    alpha = (1 - ci) / 2
    low, high = np.quantile(boots, [alpha, 1 - alpha])
    return (mean, float(low), float(high))

def _fmt_ci(mean, low, high, fmt="{:.4f}"):
    """
    Format CI as 'mean ± halfwidth' (95% CI), without changing how CIs are computed.
    Previously: 'mean [low, high]'.
    """
    if any(isinstance(v, float) and math.isnan(v) for v in (mean, low, high)):
        return "NaN"
    halfwidth = abs(high - low) / 2.0
    return f"{fmt.format(mean)} ± {fmt.format(halfwidth)}"

# ------------------------------------------------------------
# Holm–Bonferroni & effect size (for stats tests in Module 7)
# ------------------------------------------------------------
def holm_bonferroni_correction(pairs, alpha=0.05):
    """
    Holm–Bonferroni step-down correction.
    pairs: list of (name, pvalue). Returns dict[name] = (p_adj, is_significant).
    """
    m = len(pairs)
    sorted_pairs = sorted(pairs, key=lambda x: x[1])  # by p asc
    # running max of adjusted p-values to enforce monotonicity
    adj_running = {}
    running_max = 0.0
    for i, (name, p) in enumerate(sorted_pairs, start=1):
        padj = (m - i + 1) * p
        running_max = max(running_max, padj)
        adj_running[name] = running_max
    out = {}
    for name, p in pairs:
        padj = min(1.0, adj_running[name])
        out[name] = (padj, padj <= alpha)
    return out

def rank_biserial_from_wilcoxon(statistic, n):
    """
    Rank-biserial correlation for Wilcoxon signed-rank:
        RBC = 2*W / (n*(n+1)) - 1
    """
    if n <= 0 or (isinstance(statistic, float) and math.isnan(statistic)):
        return math.nan
    return float(2.0 * statistic / (n * (n + 1)) - 1.0)

# ------------------------------------------------------------
# Normalized Surface Dice (NSD)
# ------------------------------------------------------------
def _ndimg_struct_from_connectivity(connectivity: int):
    """
    Map 26/18/6 connectivity to a suitable 3D binary structure element.
    """
    if connectivity >= 26:
        return ndi.generate_binary_structure(rank=3, connectivity=3)
    if connectivity >= 18:
        return ndi.generate_binary_structure(rank=3, connectivity=2)
    return ndi.generate_binary_structure(rank=3, connectivity=1)

def _binary_surface(mask_bool: np.ndarray, struct) -> np.ndarray:
    """
    Surface voxels of a binary mask: surface = mask & ~erode(mask).
    """
    if not mask_bool.any():
        return mask_bool
    eroded = ndi.binary_erosion(mask_bool, structure=struct, iterations=1, border_value=0)
    return mask_bool & (~eroded)

def compute_nsd(pred_bin: np.ndarray,
                gt_bin:   np.ndarray,
                spacing=(1, 1, 1),
                nsd_tol_mm: float = NSD_TOL_MM,
                connectivity: int = CONNECTIVITY) -> float:
    """
    Compute NSD between binary predictions and ground truth.
      - If GT is empty: return NaN.
      - If Pred is empty but GT is not: return 0.0.
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)
    if not gt_bin.any():
        return float("nan")

    struct = _ndimg_struct_from_connectivity(connectivity)
    surf_p = _binary_surface(pred_bin, struct)
    surf_g = _binary_surface(gt_bin,   struct)

    if not pred_bin.any():
        return 0.0

    # EDT on complement of surfaces; sample distances at other surface
    dt_g = ndi.distance_transform_edt(~surf_g, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~surf_p, sampling=spacing)

    d_p2g = dt_g[surf_p]
    d_g2p = dt_p[surf_g]

    denom = int(surf_p.sum() + surf_g.sum())
    if denom == 0:
        return 1.0  # degenerate protection (consistent with typical handling)
    nsd = float(((d_p2g <= nsd_tol_mm).sum() + (d_g2p <= nsd_tol_mm).sum()) / denom)
    return nsd

# ------------------------------------------------------------
# Runtime containers (used by Modules 6–7)
# ------------------------------------------------------------
dice_metric          = DiceMetric(include_background=False, reduction="mean")
per_variant_records  = defaultdict(list)  # per positive subject
total_fp_per_variant = defaultdict(int)   # sum of FP across positive subjects
num_pos_per_variant  = defaultdict(int)   # number of positive subjects per variant

# %% [markdown] cell 14
# ## Module 6: Inference

# %% cell 15
## Module 6
## Load Models

# --- Deep Supervision wrapper (generic, supports out_channels=2 or 3) ---
class SwinUNETR_DS(nn.Module):
    """
    Swin-UNETR wrapper used ONLY for loading DS checkpoints.
    The checkpoint state_dict keys are prefixed with 'base.'.

    Note:
      - out_channels can be 2 (seg-only) or 3 (seg + DM) depending on the variant.
      - In inference we will use predictor = model.base (finest output).
    """
    def __init__(self, in_channels=3, out_channels=3, feature_size=48, scales=(0.125, 0.25, 0.5, 1.0)):
        super().__init__()
        self.base = SwinUNETR(
            in_channels=in_channels, out_channels=out_channels,
            feature_size=feature_size, use_checkpoint=True, use_v2=True
        )
        self.scales = scales

    def load_from(self, weights):
        self.base.load_from(weights)

    def forward(self, x):
        fine = self.base(x)
        outs = []
        for s in self.scales:
            outs.append(
                fine if s == 1.0 else
                F.interpolate(
                    fine, scale_factor=s, mode="trilinear",
                    align_corners=False, recompute_scale_factor=True
                )
            )
        return outs


# --- State dict helpers ---
def _unwrap_state_dict(raw):
    """
    Make state-dict robust to:
      • checkpoints saved as {'state_dict': ...}
      • DataParallel prefix 'module.'.
    """
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if len(state) > 0:
        k0 = next(iter(state.keys()))
        if k0.startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
    return state

def _is_ds_state_dict(state_dict):
    """
    Detect DS wrapper checkpoints (keys start with 'base.').
    """
    if len(state_dict) == 0:
        return False
    k0 = next(iter(state_dict.keys()))
    return k0.startswith("base.")

def _resolve_ckpt_path(ckpt_dir: Path, fold_id: int) -> Path:
    """
    Resolve checkpoint path robustly.
    Tries 'fold{fold}_best.pth', otherwise fall back to first glob match.
    """
    target = ckpt_dir / f"fold{fold_id}_best.pth"
    if target.exists():
        return target
    cands = sorted(glob.glob(str(ckpt_dir / f"*fold{fold_id}*best*.pth")))
    if cands:
        return Path(cands[0])
    raise FileNotFoundError(f"No checkpoint found for fold {fold_id} in {ckpt_dir}")

def _load_variant_model(ckpt_path: Path, out_channels: int, is_ds: bool):
    """
    Load model and return (model_for_bookkeeping, predictor_for_inference).
      - Non-DS: model = SwinUNETR(...), predictor = model
      - DS    : model = SwinUNETR_DS(...), predictor = model.base (finest scale)
               IMPORTANT: DS out_channels may be 2 (seg-only) or 3 (seg+DM).
    """
    raw = torch.load(ckpt_path, map_location=device)
    state = _unwrap_state_dict(raw)

    # If a DS checkpoint is detected, ignore `is_ds` hint and use DS wrapper.
    use_ds = _is_ds_state_dict(state) or is_ds

    if use_ds:
        model = SwinUNETR_DS(in_channels=3, out_channels=out_channels, feature_size=48)
        model.load_state_dict(state, strict=True)
        predictor = model.base
    else:
        model = SwinUNETR(
            in_channels=3, out_channels=out_channels, feature_size=48,
            use_checkpoint=True, use_v2=True
        )
        model.load_state_dict(state, strict=True)
        predictor = model

    model.to(device).eval()
    return model, predictor

# %% cell 16
## Module 6
## Inference Loop

# ------------------------------
# Inference loop (per variant)
# ------------------------------
ACTIVE_VARIANT_DIRS = OrderedDict(list(VARIANT_DIRS.items())[:MAX_VARIANTS])

for key, ckpt_dir in ACTIVE_VARIANT_DIRS.items():
    print(f"\n=== Evaluating variant: {key} ===")
    out_channels = int(VARIANT_CFG[key]["out_channels"])
    is_ds_hint   = bool(VARIANT_CFG[key].get("is_ds", False))

    for fold_id, split in enumerate(folds[:MAX_FOLDS], 1):
        ckpt_path = _resolve_ckpt_path(ckpt_dir, fold_id)
        print(f"Fold {fold_id}: loading {ckpt_path.name}")

        # ---- Load model ----
        model, predictor = _load_variant_model(ckpt_path, out_channels, is_ds_hint)

        # ---- Validation loader ----
        val_files = split["val"][:MAX_VAL_SUBJECTS] if MAX_VAL_SUBJECTS else split["val"]
        val_loader = make_dataloaders(None, val_files)

        # ---- Evaluate ----
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 1):
                if MAX_VAL_BATCHES and batch_idx > MAX_VAL_BATCHES:
                    break
                sid   = batch["sid"][0]  # preserved by MONAI dict transforms
                gt_np = batch["mask"].cpu().numpy().squeeze()  # (Z,Y,X)
                gt_pos = bool(gt_np.any())
                lbl_gt = connected_components_3d(gt_np, CONNECTIVITY, 1)

                # Forward pass
                x = torch.cat([batch[k].to(device) for k in IMG_KEYS], dim=1)
                with autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                              enabled=(device.type == "cuda")):
                    logits = sliding_window_inference(
                        x, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE,
                        overlap=OVERLAP, predictor=predictor
                    )

                # Use first two channels for segmentation (ignore DM channel if present)
                seg_logits = logits[:, :2, ...] if logits.shape[1] >= 2 else logits
                prob_np    = post_softmax(seg_logits)[:, 1].float().cpu().numpy().squeeze()

                # Positive-subject metrics only (as per your summary)
                if gt_pos:
                    # Binary prediction at 0.5 for segmentation & lesion-level detection
                    pred_bin = (prob_np > 0.5).astype(np.float32)
                    lbl_pred = connected_components_3d(pred_bin, CONNECTIVITY, MIN_VOX_PRED)

                    # Dice
                    dice_metric.reset()
                    dice_metric(
                        torch.from_numpy(pred_bin[None, None]).to(device),
                        torch.from_numpy(gt_np   [None, None]).to(device)
                    )
                    dsc_val = float(dice_metric.aggregate().cpu())

                    # NSD
                    nsd_val = compute_nsd(pred_bin, gt_np, spacing=SPACING,
                                          nsd_tol_mm=NSD_TOL_MM, connectivity=CONNECTIVITY)

                    # Lesion-level counts @ 0.5 → precision/recall/F1
                    tp, fp, fn = lesion_counts(lbl_pred, lbl_gt)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                    per_variant_records[key].append(
                        {"sid": sid, "dice": dsc_val, "nsd": nsd_val,
                         "prec": prec, "rec": rec, "f1": f1, "fp": int(fp)}
                    )
                    total_fp_per_variant[key] += int(fp)
                    num_pos_per_variant[key]  += 1

        # cleanup
        del model
        torch.cuda.empty_cache(); gc.collect()

# %% [markdown] cell 17
# ## Module 7: Summary Table

# %% cell 18
## Module 7
## Summary Table

# -------- Summary table (positive subjects only) --------
summary_rows = []
for key in ACTIVE_VARIANT_DIRS:
    recs = per_variant_records[key]

    dsc  = [r["dice"] for r in recs]
    nsd  = [r.get("nsd", np.nan) for r in recs]
    rec_ = [r["rec"]  for r in recs]
    pre_ = [r["prec"] for r in recs]
    f1_  = [r["f1"]   for r in recs]
    fp_  = [r["fp"]   for r in recs]  # FP per positive subject

    # 95% bootstrap CI on means
    dsc_m, dsc_l, dsc_h = bootstrap_ci_mean(dsc)
    nsd_m, nsd_l, nsd_h = bootstrap_ci_mean(nsd)
    rec_m, rec_l, rec_h = bootstrap_ci_mean(rec_)
    pre_m, pre_l, pre_h = bootstrap_ci_mean(pre_)
    f1_m,  f1_l,  f1_h  = bootstrap_ci_mean(f1_)
    fp_m,  fp_l,  fp_h  = bootstrap_ci_mean(fp_)

    summary_rows.append({
        "Model":                   key,
        "DSC (mean±95%CI)":        _fmt_ci(dsc_m, dsc_l, dsc_h),
        "NSD (mean±95%CI)":        _fmt_ci(nsd_m, nsd_l, nsd_h),
        "Recall (mean±95%CI)":     _fmt_ci(rec_m, rec_l, rec_h),
        "Precision (mean±95%CI)":  _fmt_ci(pre_m, pre_l, pre_h),
        "F1 (mean±95%CI)":         _fmt_ci(f1_m,  f1_l,  f1_h),
        "FP/subject (mean±95%CI)": _fmt_ci(fp_m,  fp_l,  fp_h)
    })

tbl = pd.DataFrame(summary_rows)
print("\n================  Macro Per-Case Results (Positive Subjects Only)  ================")
print(tbl.to_string(index=False))

# %% cell 19
## Module 7
## Stats Tests

# -------- Wilcoxon signed-rank tests (paired, positive subjects only) --------
metrics_for_wilcoxon = [
    ("dice", True),   # higher is better
    ("nsd",  True),
    ("rec",  True),
    ("prec", True),
    ("f1",   True),
    ("fp",   False),  # lower is better
]
BASELINE_KEY = "A0"

def _records_to_series_map(metric_key):
    """
    Build {variant: {sid: value}} for paired testing.
    Drops NaNs to allow fair overlap of subjects across variants.
    """
    out = {}
    for v in ACTIVE_VARIANT_DIRS.keys():
        m = {}
        for r in per_variant_records[v]:
            val = r.get(metric_key, np.nan)
            if not (isinstance(val, float) and math.isnan(val)):
                m[r["sid"]] = float(val)
        out[v] = m
    return out

def wilcoxon_stepwise(metric_key, higher_is_better=True):
    """
    Adjacent comparisons along VARIANT_DIRS order (e.g., A0→A1, A1→A2, ...).
    Returns DataFrame with p-values (Holm–Bonferroni corrected), median deltas, and effect size (RBC).
    """
    s_map = _records_to_series_map(metric_key)
    variants = list(ACTIVE_VARIANT_DIRS.keys())
    rows = []
    pairs_for_correction = []

    for i in range(len(variants) - 1):
        a, b = variants[i], variants[i + 1]
        sids = sorted(set(s_map[a].keys()) & set(s_map[b].keys()))
        if len(sids) == 0:
            rows.append({
                "pair": f"{a}→{b}", "metric": metric_key, "n": 0,
                "statistic": math.nan, "p_value": math.nan, "p_adj": math.nan,
                "median_delta": math.nan, "direction": "NA", "rbc": math.nan
            })
            continue

        x = np.asarray([s_map[a][s] for s in sids], float)
        y = np.asarray([s_map[b][s] for s in sids], float)
        try:
            diff = y - x
            res  = wilcoxon(diff, zero_method="pratt", alternative="two-sided", method="auto")
            stat, p = float(res.statistic), float(res.pvalue)
        except ValueError:
            stat, p = (math.nan, 1.0)

        delta = float(np.median(y - x))
        good  = (delta > 0) if higher_is_better else (delta < 0)
        direction = "↑" if good else ("↓" if delta != 0 else "→")
        rbc = rank_biserial_from_wilcoxon(stat, len(sids))

        rows.append({
            "pair": f"{a}→{b}", "metric": metric_key, "n": int(len(sids)),
            "statistic": stat, "p_value": p, "p_adj": None,
            "median_delta": delta, "direction": direction, "rbc": rbc
        })
        pairs_for_correction.append((f"{a}→{b}", p))

    if pairs_for_correction:
        adj_map = holm_bonferroni_correction(pairs_for_correction, alpha=0.05)
        for r in rows:
            nm = r["pair"]
            if nm in adj_map:
                padj, sig = adj_map[nm]
                r["p_adj"] = padj
                r["significant"] = "Yes" if sig else "No"
            else:
                r["p_adj"] = math.nan
                r["significant"] = "NA"
    else:
        for r in rows:
            r["significant"] = "NA"
            r["p_adj"] = math.nan

    return pd.DataFrame(rows)

def wilcoxon_vs_baseline(metric_key, baseline_key=BASELINE_KEY, higher_is_better=True):
    """
    Comparisons vs baseline (baseline_key → each other variant).
    Returns DataFrame with p-values (Holm–Bonferroni corrected), median deltas, and RBC.
    """
    s_map = _records_to_series_map(metric_key)
    variants = [v for v in ACTIVE_VARIANT_DIRS.keys() if v != baseline_key]
    rows = []
    pairs_for_correction = []

    for b in variants:
        a = baseline_key
        sids = sorted(set(s_map[a].keys()) & set(s_map[b].keys()))
        if len(sids) == 0:
            rows.append({
                "pair": f"{a}→{b}", "metric": metric_key, "n": 0,
                "statistic": math.nan, "p_value": math.nan, "p_adj": math.nan,
                "median_delta": math.nan, "direction": "NA", "rbc": math.nan
            })
            continue

        x = np.asarray([s_map[a][s] for s in sids], float)
        y = np.asarray([s_map[b][s] for s in sids], float)
        try:
            diff = y - x
            res  = wilcoxon(diff, zero_method="pratt", alternative="two-sided", method="auto")
            stat, p = float(res.statistic), float(res.pvalue)
        except ValueError:
            stat, p = (math.nan, 1.0)

        delta = float(np.median(y - x))
        good  = (delta > 0) if higher_is_better else (delta < 0)
        direction = "↑" if good else ("↓" if delta != 0 else "→")
        rbc = rank_biserial_from_wilcoxon(stat, len(sids))

        rows.append({
            "pair": f"{a}→{b}", "metric": metric_key, "n": int(len(sids)),
            "statistic": stat, "p_value": p, "p_adj": None,
            "median_delta": delta, "direction": direction, "rbc": rbc
        })
        pairs_for_correction.append((f"{a}→{b}", p))

    if pairs_for_correction:
        adj_map = holm_bonferroni_correction(pairs_for_correction, alpha=0.05)
        for r in rows:
            nm = r["pair"]
            if nm in adj_map:
                padj, sig = adj_map[nm]
                r["p_adj"] = padj
                r["significant"] = "Yes" if sig else "No"
            else:
                r["p_adj"] = math.nan
                r["significant"] = "NA"
    else:
        for r in rows:
            r["significant"] = "NA"
            r["p_adj"] = math.nan

    return pd.DataFrame(rows)

# Aggregate tests
stepwise_dfs = []
vsbase_dfs   = []
for m, hib in metrics_for_wilcoxon:
    stepwise_dfs.append(wilcoxon_stepwise(m, higher_is_better=hib))
    vsbase_dfs.append(wilcoxon_vs_baseline(m, baseline_key=BASELINE_KEY, higher_is_better=hib))

wilcoxon_stepwise_all = pd.concat(stepwise_dfs, axis=0, ignore_index=True)
wilcoxon_vsbase_all   = pd.concat(vsbase_dfs,   axis=0, ignore_index=True)

stepwise_chain = "→".join(list(ACTIVE_VARIANT_DIRS.keys()))
vsbase_pairs   = ", ".join([f"{BASELINE_KEY}→{v}" for v in ACTIVE_VARIANT_DIRS.keys() if v != BASELINE_KEY])

print(f"\n================  Wilcoxon (Stepwise: {stepwise_chain}) with Holm–Bonferroni  ================")
print(wilcoxon_stepwise_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\n================  Wilcoxon (vs Baseline {BASELINE_KEY}: {vsbase_pairs}) with Holm–Bonferroni  ========================")
print(wilcoxon_vsbase_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
