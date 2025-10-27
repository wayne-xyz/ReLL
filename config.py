from __future__ import annotations

"""
Global configuration dataclasses and helpers for the ReLL training pipeline.

The defaults mirror the values that previously lived at the bottom of
Train/Train.py so the CLI can stay lightweight while allowing overrides.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    sample_root: Sequence[Path]
    lidar_variant: str = "gicp"
    cache_rasters: bool = True
    max_translation_px: int = 0
    max_rotation_deg: float = 0.0
    target_sigma_xy_m: float = 0.2
    target_sigma_yaw_rad: float = math.radians(1.0)
    raster_builder: Optional[Callable[[Path], Dict[str, Any]]] = None


@dataclass
class ModelConfig:
    lidar_in_channels: int = 3
    map_in_channels: int = 5
    embed_dim: int = 128
    encoder_depth: int = 4
    search_radius: int = 8
    theta_bins: int = 25
    theta_range_deg: float = 4.0
    height_attention: bool = True
    height_recon_weight: float = 0.0
    proj_dim: int = 4
    w_xy: float = 1.0
    w_theta: float = 0.5
    alpha_xy: float = 0.04
    alpha_theta: float = 0.20


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SaveConfig:
    enable: bool = True
    save_dir: Union[str, Path] = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    filename_best: str = "best.ckpt"
    filename_last: str = "last.ckpt"


@dataclass
class EarlyStopConfig:
    enabled: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    min_delta: float = 1e-3
    patience: int = 5
    warmup_epochs: int = 5
    restore_best: bool = True


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------

DEFAULT_SAVE_ROOT = Path("/content/drive/MyDrive/Rell-model")
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Rell-sample-raster")


def default_dataset_config() -> DatasetConfig:
    return DatasetConfig(
        sample_root=[],
        lidar_variant="gicp",
        cache_rasters=True,
        max_translation_px=8,
        max_rotation_deg=0.0,
        target_sigma_xy_m=0.3,
        target_sigma_yaw_rad=math.radians(0.5),
        raster_builder=None,
    )


def default_model_config() -> ModelConfig:
    return ModelConfig(
        lidar_in_channels=3,
        map_in_channels=5,
        embed_dim=32,
        encoder_depth=4,
        search_radius=8,
        theta_bins=25,
        theta_range_deg=4.0,
        height_attention=True,
        proj_dim=4,
    )


def default_optim_config() -> OptimConfig:
    return OptimConfig(
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=16,
        num_workers=2,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def default_save_config(save_dir: Optional[Union[str, Path]] = None) -> SaveConfig:
    return SaveConfig(
        enable=True,
        save_dir=save_dir if save_dir is not None else DEFAULT_SAVE_ROOT,
        save_best=True,
        save_last=True,
        monitor="val_loss",
        mode="min",
        filename_best="best_1000.ckpt",
        filename_last="last_1000.ckpt",
    )


def default_early_stop_config() -> EarlyStopConfig:
    return EarlyStopConfig(
        enabled=True,
        monitor="val_loss",
        mode="min",
        min_delta=1e-3,
        patience=10,
        warmup_epochs=20,
        restore_best=True,
    )


def load_default_configs(
    *,
    save_dir: Optional[Union[str, Path]] = None,
) -> Tuple[DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig]:
    """
    Returns fresh instances of all configs so that downstream code can mutate
    them without accidentally sharing state across runs.
    """

    return (
        default_dataset_config(),
        default_model_config(),
        default_optim_config(),
        default_save_config(save_dir),
        default_early_stop_config(),
    )


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "OptimConfig",
    "SaveConfig",
    "EarlyStopConfig",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SAVE_ROOT",
    "default_dataset_config",
    "default_model_config",
    "default_optim_config",
    "default_save_config",
    "default_early_stop_config",
    "load_default_configs",
]
