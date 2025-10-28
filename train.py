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
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay.")
    parser.add_argument("--search-radius", type=int, default=None, help="Override model search radius in pixels.")
    parser.add_argument("--embed-dim", type=int, default=None, help="Override model embedding dimension.")
    parser.add_argument("--proj-dim", type=int, default=None, help="Override projection dimension.")
    parser.add_argument("--encoder-depth", type=int, default=None, help="Override encoder depth.")
    parser.add_argument("--stem-channels", type=int, default=None, help="Override stem channels.")
    parser.add_argument(
        "--subset-frac",
        type=float,
        default=1.0,
        help="Optional fraction of samples to use for quick experiments (0 < frac <= 1).",
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

    if args.embed_dim is not None:
        model_cfg.embed_dim = args.embed_dim
    if args.proj_dim is not None:
        model_cfg.proj_dim = args.proj_dim
    if args.encoder_depth is not None:
        model_cfg.encoder_depth = args.encoder_depth
    if args.stem_channels is not None:
        model_cfg.stem_channels = args.stem_channels
    if args.search_radius is not None:
        model_cfg.search_radius = args.search_radius

    dataset_cfg.max_translation_px = model_cfg.search_radius
    dataset_cfg.max_rotation_deg = 0.0

    optim_cfg.device = _detect_device(args.device)
    if args.batch_size is not None:
        optim_cfg.batch_size = args.batch_size
    if args.epochs is not None:
        optim_cfg.epochs = args.epochs
    if args.lr is not None:
        optim_cfg.lr = args.lr
    if args.weight_decay is not None:
        optim_cfg.weight_decay = args.weight_decay

    save_cfg.monitor = "val_pixel_error"
    save_cfg.mode = "min"
    early_cfg.monitor = "val_pixel_error"
    early_cfg.mode = "min"

    print(f"[Device] Training on: {optim_cfg.device}")

    data_root = args.data_root or DEFAULT_DATA_ROOT
    train_localization_model(
        processed_raster_data_dir=data_root,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        save_cfg=save_cfg,
        early_stop_cfg=early_cfg,
        subset_fraction=args.subset_frac,
    )


if __name__ == "__main__":
    main()
