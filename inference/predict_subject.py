#!/usr/bin/env python
"""Run lacune segmentation for one subject from the command line."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)


IMAGE_KEYS = ("flair", "t1", "t2")


class SwinUNETRDS(nn.Module):
    """Deep-supervision wrapper used for checkpoints with `base.` keys."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_size: int = 48,
        scales: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0),
    ) -> None:
        super().__init__()
        self.base = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )
        self.scales = scales

    def load_from(self, weights):
        self.base.load_from(weights)

    def forward(self, x):
        fine = self.base(x)
        outs = []
        for scale in self.scales:
            if scale == 1.0:
                outs.append(fine)
            else:
                outs.append(
                    F.interpolate(
                        fine,
                        scale_factor=scale,
                        mode="trilinear",
                        align_corners=False,
                        recompute_scale_factor=True,
                    )
                )
        return outs


def unwrap_state_dict(raw):
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if state:
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def is_deep_supervision_state(state_dict) -> bool:
    return bool(state_dict) and next(iter(state_dict.keys())).startswith("base.")


def infer_out_channels(state_dict) -> int:
    preferred_suffixes = (
        "out.conv.conv.weight",
        "out.conv.weight",
        "output_block.conv.conv.weight",
    )
    for suffix in preferred_suffixes:
        for key, value in state_dict.items():
            clean_key = key[len("base.") :] if key.startswith("base.") else key
            if clean_key.endswith(suffix) and hasattr(value, "shape"):
                return int(value.shape[0])

    candidates = []
    for key, value in state_dict.items():
        clean_key = key[len("base.") :] if key.startswith("base.") else key
        if "out" in clean_key and hasattr(value, "shape") and len(value.shape) >= 1:
            if int(value.shape[0]) in (2, 3):
                candidates.append(int(value.shape[0]))
    if candidates:
        return candidates[-1]

    raise ValueError(
        "Could not infer out_channels from the checkpoint. "
        "Pass --out-channels 2 or --out-channels 3."
    )


def load_model(checkpoint_path: Path, out_channels: int | None, feature_size: int, device: torch.device):
    raw = torch.load(checkpoint_path, map_location=device)
    state = unwrap_state_dict(raw)
    out_channels = out_channels or infer_out_channels(state)
    use_ds = is_deep_supervision_state(state)

    if use_ds:
        model = SwinUNETRDS(in_channels=3, out_channels=out_channels, feature_size=feature_size)
        model.load_state_dict(state, strict=True)
        predictor = model.base
    else:
        model = SwinUNETR(
            in_channels=3,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )
        model.load_state_dict(state, strict=True)
        predictor = model

    model.to(device).eval()
    return model, predictor, out_channels, use_ds


def build_preprocess(spacing: tuple[float, float, float]):
    return Compose(
        [
            LoadImaged(keys=IMAGE_KEYS),
            EnsureChannelFirstd(keys=IMAGE_KEYS),
            Orientationd(keys=IMAGE_KEYS, axcodes="RAS"),
            Spacingd(
                keys=IMAGE_KEYS,
                pixdim=spacing,
                mode=("trilinear", "trilinear", "trilinear"),
                align_corners=True,
            ),
            NormalizeIntensityd(keys=IMAGE_KEYS),
            EnsureTyped(keys=IMAGE_KEYS),
        ]
    )


def get_affine(meta_tensor) -> np.ndarray:
    affine = meta_tensor.meta.get("affine")
    if affine is None:
        return np.eye(4, dtype=np.float64)
    if hasattr(affine, "detach"):
        affine = affine.detach().cpu().numpy()
    return np.asarray(affine, dtype=np.float64)


def remove_small_components(mask: np.ndarray, min_voxels: int, connectivity: int = 26) -> np.ndarray:
    if min_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)

    import cc3d

    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=connectivity)
    keep = np.zeros_like(mask, dtype=bool)
    component_ids, counts = np.unique(labels, return_counts=True)
    for component_id, count in zip(component_ids, counts):
        if component_id != 0 and count >= min_voxels:
            keep |= labels == component_id
    return keep.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained lacune segmentation checkpoint on one subject."
    )
    parser.add_argument("--flair", required=True, type=Path, help="Path to the FLAIR NIfTI file.")
    parser.add_argument("--t1", required=True, type=Path, help="Path to the T1 NIfTI file.")
    parser.add_argument("--t2", required=True, type=Path, help="Path to the T2 NIfTI file.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a fold checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Output binary mask NIfTI path.")
    parser.add_argument(
        "--probability-output",
        type=Path,
        default=None,
        help="Optional output path for the foreground probability map.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Foreground threshold.")
    parser.add_argument("--min-voxels", type=int, default=5, help="Remove predicted components below this size.")
    parser.add_argument("--out-channels", type=int, choices=(2, 3), default=None, help="Override checkpoint output channels.")
    parser.add_argument("--feature-size", type=int, default=48, help="SwinUNETR feature size.")
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128), help="Sliding-window ROI size.")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Inference spacing in mm.")
    parser.add_argument("--sw-batch-size", type=int, default=1, help="Sliding-window batch size.")
    parser.add_argument("--overlap", type=float, default=0.6, help="Sliding-window overlap.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    preprocess = build_preprocess(tuple(args.spacing))
    data = preprocess({key: str(getattr(args, key)) for key in IMAGE_KEYS})
    image = torch.cat([data[key].unsqueeze(0).to(device) for key in IMAGE_KEYS], dim=1)
    affine = get_affine(data["flair"])

    model, predictor, out_channels, use_ds = load_model(
        args.checkpoint,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        device=device,
    )

    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        logits = sliding_window_inference(
            image,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            predictor=predictor,
        )

    seg_logits = logits[:, :2, ...]
    probability = torch.softmax(seg_logits, dim=1)[0, 1].float().cpu().numpy()
    mask = remove_small_components((probability > args.threshold).astype(np.uint8), args.min_voxels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(args.output))

    if args.probability_output is not None:
        args.probability_output.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(probability.astype(np.float32), affine), str(args.probability_output))

    print(f"Saved prediction mask: {args.output}")
    if args.probability_output is not None:
        print(f"Saved foreground probability map: {args.probability_output}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Detected out_channels={out_channels}, deep_supervision={use_ds}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
