from __future__ import annotations

"""
Global configuration dataclasses and helpers for the ReLL training pipeline.

Configuration values are sourced from YAML (Train/default.yaml by default)
and validated against the typed schema defined in this module. CLI overrides
layer on top of the YAML values so there is a single source of truth.
"""

import copy
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")
_FALLBACK_DATA_ROOT = Path("/content/drive/MyDrive/Rell-sample-raster")
_FALLBACK_SAVE_ROOT = Path("/content/drive/MyDrive/Rell-model")


class ConfigError(RuntimeError):
    """Raised when configuration files are malformed or inconsistent."""


def _read_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping at top level of {path}")
    return data


def _first_sample_root(snapshot: Dict[str, Any]) -> Optional[Path]:
    raw_roots = snapshot.get("dataset", {}).get("sample_root")
    if raw_roots is None:
        return None
    if isinstance(raw_roots, (str, Path)):
        return Path(raw_roots)
    if isinstance(raw_roots, (list, tuple)) and raw_roots:
        return Path(raw_roots[0])
    return None


try:
    _DEFAULT_CONFIG_SNAPSHOT = _read_yaml_file(DEFAULT_CONFIG_PATH)
except (FileNotFoundError, ConfigError):
    _DEFAULT_CONFIG_SNAPSHOT = {}

_DEFAULT_SAMPLE_ROOT = _first_sample_root(_DEFAULT_CONFIG_SNAPSHOT)
DEFAULT_DATA_ROOT = _DEFAULT_SAMPLE_ROOT or _FALLBACK_DATA_ROOT

_DEFAULT_SAVE_DIR = _DEFAULT_CONFIG_SNAPSHOT.get("save", {}).get("save_dir")
DEFAULT_SAVE_ROOT = Path(_DEFAULT_SAVE_DIR) if _DEFAULT_SAVE_DIR else _FALLBACK_SAVE_ROOT


@dataclass
class DatasetConfig:
    sample_root: Tuple[Path, ...]
    lidar_variant: str
    cache_rasters: bool
    max_translation_px: int
    max_rotation_deg: float
    train_fraction: float
    raster_builder: Optional[Callable[[Path], Dict[str, Any]]] = None


@dataclass
class ModelConfig:
    lidar_in_channels: int
    map_in_channels: int
    stem_channels: int
    embed_dim: int
    encoder_depth: int
    encoder_base_channels: int
    proj_dim: int
    search_radius: int
    theta_search_deg: int
    translation_smoothing_kernel: int
    gaussian_sigma_px: float
    w_xy: float
    w_theta: float
    gaussian_sigma_theta_deg: float
    sigma_weight_xy: float
    sigma_weight_theta: float


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    batch_size: int
    num_workers: int
    epochs: int
    device: str


@dataclass
class SaveConfig:
    enable: bool
    save_dir: Path
    save_best: bool
    save_last: bool
    monitor: str
    mode: str
    filename_best: str
    filename_last: str


@dataclass
class EarlyStopConfig:
    enabled: bool
    monitor: str
    mode: str
    min_delta: float
    patience: int
    warmup_epochs: int
    restore_best: bool


_CONFIG_SECTION_TYPES = {
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "save": SaveConfig,
    "early_stop": EarlyStopConfig,
}


def _merge_nested(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            base_value = merged.get(key)
            if isinstance(base_value, Mapping):
                merged[key] = _merge_nested(base_value, value)  # type: ignore[arg-type]
            else:
                merged[key] = _merge_nested({}, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_section(section_name: str, config: Mapping[str, Any]) -> Any:
    section_type = _CONFIG_SECTION_TYPES[section_name]
    payload = config.get(section_name)
    if payload is None:
        raise ConfigError(f"Missing '{section_name}' section in configuration")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Section '{section_name}' must be a mapping")

    declared_fields = {field.name: field for field in fields(section_type)}
    values: Dict[str, Any] = {}
    for field_name, field_def in declared_fields.items():
        if field_name in payload:
            values[field_name] = _coerce_value(field_name, payload[field_name])
        elif field_def.default is not MISSING:
            values[field_name] = field_def.default
        elif getattr(field_def, "default_factory", MISSING) is not MISSING:
            values[field_name] = field_def.default_factory()  # type: ignore[misc]
        else:
            raise ConfigError(f"Missing key '{field_name}' in '{section_name}' configuration")

    unknown = set(payload.keys()) - set(declared_fields.keys())
    if unknown:
        raise ConfigError(f"Unknown key(s) {sorted(unknown)} in '{section_name}' configuration")

    return section_type(**values)


def _coerce_value(field_name: str, value: Any) -> Any:
    if field_name == "sample_root":
        if value is None:
            return ()
        if isinstance(value, (str, Path)):
            return (Path(value),)
        return tuple(Path(item) for item in value)
    if field_name.endswith("_dir") or field_name.endswith("_path"):
        return Path(value)
    if isinstance(value, Path):
        return value
    return value


def load_default_configs(
    *,
    save_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig]:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config_dict = _read_yaml_file(source_path)

    if overrides:
        unknown_sections = set(overrides.keys()) - set(_CONFIG_SECTION_TYPES.keys())
        if unknown_sections:
            raise ConfigError(f"Unknown config sections in overrides: {sorted(unknown_sections)}")
        config_dict = _merge_nested(config_dict, overrides)

    dataset_cfg = _build_section("dataset", config_dict)
    model_cfg = _build_section("model", config_dict)
    optim_cfg = _build_section("optim", config_dict)
    save_cfg = _build_section("save", config_dict)
    early_cfg = _build_section("early_stop", config_dict)

    if save_dir is not None:
        save_cfg = replace(save_cfg, save_dir=Path(save_dir))

    return dataset_cfg, model_cfg, optim_cfg, save_cfg, early_cfg


__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ModelConfig",
    "OptimConfig",
    "SaveConfig",
    "EarlyStopConfig",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SAVE_ROOT",
    "load_default_configs",
]
