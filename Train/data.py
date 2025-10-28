from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import DatasetConfig, OptimConfig

try:
    from torchvision.transforms.functional import InterpolationMode
    from torchvision.transforms.functional import affine as tv_affine
except Exception:  # pragma: no cover - torchvision optional
    tv_affine = None
    InterpolationMode = None


def replace_nan_with_zero(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    mask = torch.isfinite(tensor)
    cleaned = torch.where(mask, tensor, torch.zeros_like(tensor))
    return cleaned, mask.float()


def affine_warp(tensor: Tensor, angle_deg: float, translate_px: Tuple[float, float]) -> Tensor:
    if tv_affine is None:
        if abs(angle_deg) > 1e-6 or any(abs(t) > 1e-6 for t in translate_px):
            raise RuntimeError(
                "Rotation/translation augmentation requires torchvision. "
                "Install torchvision or set max_rotation_deg=0 and max_translation_px=0.",
            )
        return tensor
    return tv_affine(
        tensor,
        angle=angle_deg,
        translate=translate_px,
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Imagery must be 3D, got shape {arr.shape}")
    if arr.shape[-1] in (1, 3, 4, 5, 6, 7, 8) and arr.shape[0] not in (1, 3, 4, 5, 6, 7, 8):
        return np.transpose(arr, (2, 0, 1))
    return arr


def raster_builder_from_processed_dir(sample_dir: Path) -> Dict[str, np.ndarray]:
    def load_npy(name: str) -> np.ndarray:
        p = sample_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        return np.load(p, mmap_mode="r")

    rasters = {
        "gicp_height": load_npy("gicp_height.npy"),
        "gicp_intensity": load_npy("gicp_intensity.npy"),
        "non_aligned_height": load_npy("non_aligned_height.npy"),
        "non_aligned_intensity": load_npy("non_aligned_intensity.npy"),
        "dsm_height": load_npy("dsm_height.npy"),
        "imagery": _ensure_chw(load_npy("imagery.npy")),
    }

    res_file = sample_dir / "resolution.pkl"
    transform_file = sample_dir / "transform.pkl"
    if res_file.exists():
        with open(res_file, "rb") as f:
            resolution = float(pickle.load(f))
    elif transform_file.exists():
        with open(transform_file, "rb") as f:
            transform = pickle.load(f)
        resolution = float(abs(getattr(transform, "a", None)))
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"Invalid resolution from transform for {sample_dir}")
    else:
        raise FileNotFoundError(
            f"Neither resolution.pkl nor transform.pkl found in {sample_dir}",
        )

    rasters["resolution"] = resolution
    return rasters


class GeoAlignRasterDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sample_dirs: List[Path] = sorted(map(Path, config.sample_root))
        if not self.sample_dirs:
            raise ValueError("No sample directories were provided.")
        if config.raster_builder is None:
            raise ValueError("DatasetConfig.raster_builder must be provided.")
        self._builder = config.raster_builder
        self._cache: Dict[int, Dict[str, Tensor]] = {}
        self._meta_resolution: Dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_sample(self, idx: int) -> Dict[str, Tensor]:
        if self.config.cache_rasters and idx in self._cache:
            return self._cache[idx]

        rasters = self._builder(self.sample_dirs[idx])

        if self.config.lidar_variant.lower() == "gicp":
            lidar_height = torch.tensor(rasters["gicp_height"], dtype=torch.float32)
            lidar_intensity = torch.tensor(rasters["gicp_intensity"], dtype=torch.float32)
        elif self.config.lidar_variant.lower() in {"non_gicp", "non_aligned"}:
            lidar_height = torch.from_numpy(rasters["non_aligned_height"]).float()
            lidar_intensity = torch.from_numpy(rasters["non_aligned_intensity"]).float()
        else:
            raise ValueError("lidar_variant must be 'gicp' or 'non_aligned'.")

        dsm_height = torch.tensor(rasters["dsm_height"], dtype=torch.float32)
        imagery = torch.tensor(rasters["imagery"], dtype=torch.float32)

        if imagery.shape[0] > 3:
            imagery = imagery[:3, :, :]

        lidar_height, lidar_mask = replace_nan_with_zero(lidar_height)
        lidar_intensity, _ = replace_nan_with_zero(lidar_intensity)
        dsm_height, dsm_mask = replace_nan_with_zero(dsm_height)

        lidar_tensor = torch.stack(
            [lidar_height, lidar_intensity / 255.0, lidar_mask],
            dim=0,
        )

        if imagery.dtype == torch.uint8:
            imagery = imagery.float() / 255.0
        dsm_height = dsm_height.unsqueeze(0)
        dsm_mask = dsm_mask.unsqueeze(0)
        map_tensor = torch.cat([imagery, dsm_height, dsm_mask], dim=0)

        if idx not in self._meta_resolution:
            self._meta_resolution[idx] = float(rasters["resolution"])
        resolution = torch.tensor(self._meta_resolution[idx], dtype=torch.float32)

        sample = {"lidar": lidar_tensor, "map": map_tensor, "resolution": resolution}
        if self.config.cache_rasters:
            self._cache[idx] = sample
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        base = self._load_sample(idx)
        lidar = base["lidar"]
        geospatial = base["map"]
        resolution = base["resolution"]

        res = float(resolution.item())
        if self.config.max_translation_px and self.config.max_translation_px > 0:
            max_shift_px = int(self.config.max_translation_px)
        else:
            max_shift_px = max(1, int(round(1.0 / max(res, 1e-6))))

        rot_deg = self.config.max_rotation_deg

        dx_px = random.randint(-max_shift_px, max_shift_px)
        dy_px = random.randint(-max_shift_px, max_shift_px)
        dtheta_deg = random.uniform(-rot_deg, rot_deg)

        warped_lidar = affine_warp(lidar, angle_deg=dtheta_deg, translate_px=(dx_px, dy_px))

        dx_m = dx_px * res
        dy_m = dy_px * res
        dtheta_rad = math.radians(dtheta_deg)

        target_mu = torch.tensor([-dx_m, -dy_m, -dtheta_rad], dtype=torch.float32)
        target_sigma = torch.tensor(
            [
                self.config.target_sigma_xy_m,
                self.config.target_sigma_xy_m,
                self.config.target_sigma_yaw_rad,
            ],
            dtype=torch.float32,
        )

        return {
            "lidar": warped_lidar,
            "map": geospatial,
            "pose_mu": target_mu,
            "pose_sigma": target_sigma,
            "resolution": resolution,
            "sample_idx": torch.tensor(idx, dtype=torch.int64),
        }


def create_dataloader(dataset: GeoAlignRasterDataset, optim_cfg: OptimConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=optim_cfg.batch_size,
        shuffle=shuffle,
        num_workers=optim_cfg.num_workers,
        pin_memory=optim_cfg.device.startswith("cuda"),
        drop_last=False,
        persistent_workers=optim_cfg.num_workers > 0,
    )


__all__ = [
    "GeoAlignRasterDataset",
    "create_dataloader",
    "raster_builder_from_processed_dir",
    "replace_nan_with_zero",
    "affine_warp",
]
