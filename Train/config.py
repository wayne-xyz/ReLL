from __future__ import annotations

"""
Global configuration dataclasses and helpers for the ReLL training pipeline.

The defaults mirror the values that previously lived at the bottom of
Train/Train.py so the CLI can stay lightweight while allowing overrides.
"""

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import yaml


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
    stem_channels: int = 32
    embed_dim: int = 128
    encoder_depth: int = 3
    encoder_base_channels: int = 64
    proj_dim: int = 32
    search_radius: int = 8
    translation_smoothing_kernel: int = 1
    gaussian_sigma_px: float = 1.0
    w_xy: float = 1.0


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
        max_translation_px=4,
        max_rotation_deg=0.0,
        target_sigma_xy_m=0.3,
        target_sigma_yaw_rad=math.radians(0.5),
        raster_builder=None,
    )


def default_model_config() -> ModelConfig:
    return ModelConfig(
        lidar_in_channels=3,
        map_in_channels=5,
        stem_channels=32,
        embed_dim=64,
        encoder_depth=3,
        encoder_base_channels=64,
        proj_dim=32,
        search_radius=4,
        translation_smoothing_kernel=1,
        gaussian_sigma_px=1.0,
    )


def default_optim_config() -> OptimConfig:
    return OptimConfig(
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=8,
        num_workers=2,
        epochs=200,
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
        patience=50,
        warmup_epochs=20,
        restore_best=True,
    )


def load_default_configs(
    *,
    save_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig]:
    """
    Returns fresh instances of all configs so that downstream code can mutate
    them without accidentally sharing state across runs.
    """

    dataset_cfg = default_dataset_config()
    model_cfg = default_model_config()
    optim_cfg = default_optim_config()
    save_cfg = default_save_config()
    early_cfg = default_early_stop_config()

    overrides = _load_yaml_config(config_path)
    if overrides:
        dataset_cfg = _apply_overrides(dataset_cfg, overrides.get("dataset"))
        model_cfg = _apply_overrides(model_cfg, overrides.get("model"))
        optim_cfg = _apply_overrides(optim_cfg, overrides.get("optim"))
        save_cfg = _apply_overrides(save_cfg, overrides.get("save"))
        early_cfg = _apply_overrides(early_cfg, overrides.get("early_stop"))

    if save_dir is not None:
        save_cfg = replace(save_cfg, save_dir=Path(save_dir))

    return dataset_cfg, model_cfg, optim_cfg, save_cfg, early_cfg


def _load_yaml_config(path: Optional[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {config_path}")
    return data


def _apply_overrides(instance: Any, overrides: Optional[Dict[str, Any]]) -> Any:
    if not overrides:
        return instance
    valid_fields = {f.name for f in fields(instance)}
    updates: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise KeyError(f"Unknown config key '{key}' for {type(instance).__name__}")
        updates[key] = _coerce_value(key, value, getattr(instance, key))
    return replace(instance, **updates)


def _coerce_value(field_name: str, value: Any, current_value: Any) -> Any:
    if field_name == "sample_root":
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(v) for v in value]
    if isinstance(current_value, Path):
        return Path(value) if value is not None else None
    if field_name.endswith("_dir") or field_name.endswith("_path"):
        return Path(value) if value is not None else None
    return value


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
