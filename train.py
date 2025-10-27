from __future__ import annotations

import argparse
from pathlib import Path
import os

import torch

from config import DEFAULT_DATA_ROOT, load_default_configs
from Train.engine import train_localization_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ReLL localization model.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing processed raster samples (defaults to config.DEFAULT_DATA_ROOT).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory from config (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training device (e.g. cpu, cuda, cuda:1).",
    )
    return parser.parse_args()


def _detect_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    if torch.cuda.is_available():
        return "cuda"

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    print(
        "[Device] CUDA not available according to PyTorch. "
        f"torch.version.cuda={torch.version.cuda}, "
        f"compiled_with_cuda={torch.backends.cuda.is_built()}, "
        f"device_count={torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={cuda_visible}",
    )
    print("[Device] Falling back to CPU. Pass --device cuda to force an error if you expect GPU access.")
    return "cpu"


def main() -> None:
    args = parse_args()
    dataset_cfg, model_cfg, optim_cfg, save_cfg, early_cfg = load_default_configs(
        save_dir=args.save_dir,
    )

    optim_cfg.device = _detect_device(args.device)

    print(f"[Device] Training on: {optim_cfg.device}")

    data_root = args.data_root or DEFAULT_DATA_ROOT
    train_localization_model(
        processed_raster_data_dir=data_root,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        save_cfg=save_cfg,
        early_stop_cfg=early_cfg,
    )


if __name__ == "__main__":
    main()
