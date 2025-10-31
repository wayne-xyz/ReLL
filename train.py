from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from Train.config import DEFAULT_CONFIG_PATH, DEFAULT_DATA_ROOT, load_default_configs
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
    parser.add_argument("--max-rotation-deg", type=float, default=None, help="Override max rotation augmentation (deg).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file to override defaults (defaults to Train/default.yaml if present).",
    )
    parser.add_argument(
        "--subset-frac",
        type=float,
        default=1.0,
        help="Optional fraction of samples to use for quick experiments (0 < frac <= 1).",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Save training curves (loss/RMS metrics) to an image inside --save-dir after training.",
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
    cli_overrides: Dict[str, Dict[str, Any]] = {}

    def _set_override(section: str, key: str, value: Any) -> None:
        if value is None:
            return
        cli_overrides.setdefault(section, {})[key] = value

    _set_override("model", "embed_dim", args.embed_dim)
    _set_override("model", "proj_dim", args.proj_dim)
    _set_override("model", "encoder_depth", args.encoder_depth)
    _set_override("model", "stem_channels", args.stem_channels)
    _set_override("model", "search_radius", args.search_radius)
    if args.max_rotation_deg is not None:
        _set_override("dataset", "max_rotation_deg", float(args.max_rotation_deg))
    _set_override("optim", "batch_size", args.batch_size)
    _set_override("optim", "epochs", args.epochs)
    _set_override("optim", "lr", args.lr)
    _set_override("optim", "weight_decay", args.weight_decay)

    default_config_path = DEFAULT_CONFIG_PATH
    config_path = args.config or (default_config_path if default_config_path.exists() else None)

    dataset_cfg, model_cfg, optim_cfg, save_cfg, early_cfg = load_default_configs(
        save_dir=args.save_dir,
        config_path=config_path,
        overrides=cli_overrides or None,
    )

    if config_path is not None:
        print(f"[Config] Loaded overrides from: {Path(config_path).resolve()}")

    dataset_cfg.max_translation_px = model_cfg.search_radius
    model_cfg.theta_search_deg = max(int(round(dataset_cfg.max_rotation_deg)), 0)

    requested_device: Optional[str]
    if args.device == "auto":
        requested_device = None
    elif args.device is not None:
        requested_device = args.device
    else:
        yaml_device = getattr(optim_cfg, "device", None)
        requested_device = None if yaml_device in (None, "auto") else yaml_device
    optim_cfg.device = _detect_device(requested_device)

    save_cfg.monitor = "val_loss"
    save_cfg.mode = "min"
    early_cfg.monitor = "val_loss"
    early_cfg.mode = "min"

    print(f"[Device] Training on: {optim_cfg.device}")

    data_root = args.data_root or (dataset_cfg.sample_root[0] if dataset_cfg.sample_root else DEFAULT_DATA_ROOT)
    _, _, _, _, history = train_localization_model(
        processed_raster_data_dir=data_root,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        save_cfg=save_cfg,
        early_stop_cfg=early_cfg,
        subset_fraction=args.subset_frac,
    )

    if args.plot_metrics:
        _export_training_plots(history, save_cfg.save_dir)


def _export_training_plots(history: dict, save_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(f"[Plot] Matplotlib is required for plotting ({exc}). Install it or drop --plot-metrics.")
        return

    if not history or "train_loss" not in history:
        print("[Plot] No history available to plot.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    plots = [
        ("Loss", "train_loss", "val_loss", "Loss"),
        ("RMS X (m) - Softmax", "train_softmax_rms_x", "val_softmax_rms_x", "RMS X (m)"),
        ("RMS Y (m) - Softmax", "train_softmax_rms_y", "val_softmax_rms_y", "RMS Y (m)"),
        ("RMS Theta (rad)", "train_rms_theta", "val_rms_theta", "RMS θ (rad)"),
    ]

    for ax, (title, train_key, val_key, ylabel) in zip(axes, plots):
        train_data = history.get(train_key, [])
        val_data = history.get(val_key, [])
        if train_data:  # Only plot if data exists
            ax.plot(epochs[:len(train_data)], train_data, label="train")
        if val_data:
            ax.plot(epochs[:len(val_data)], val_data, label="val")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[len(plots):]:
        ax.axis("off")

    fig.suptitle("Training Progress", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir = Path(save_dir) if save_dir is not None else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "training_metrics.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved training metrics to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
