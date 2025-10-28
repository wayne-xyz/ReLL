from __future__ import annotations

import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torch.serialization import add_safe_globals

from config import (
    DatasetConfig,
    EarlyStopConfig,
    ModelConfig,
    OptimConfig,
    SaveConfig,
)
from .data import GeoAlignRasterDataset, create_dataloader, raster_builder_from_processed_dir
from .model import LocalizationCriterion, LocalizationModel


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class Trainer:
    def __init__(
        self,
        model: LocalizationModel,
        criterion: LocalizationCriterion,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device

    def _move_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def train_epoch(self, loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        total_metrics = {"rms_x": 0.0, "rms_y": 0.0, "pixel_error": 0.0}
        count = 0

        t_fetch = t_to_dev = t_forward = t_loss = t_backward = t_step = 0.0

        it = iter(loader)
        num_batches = len(loader)

        for _ in range(num_batches):
            t0 = time.perf_counter()
            batch = next(it)
            t1 = time.perf_counter()
            t_fetch += t1 - t0

            t2 = time.perf_counter()
            batch = self._move_batch(batch)
            _maybe_sync(self.device)
            t3 = time.perf_counter()
            t_to_dev += t3 - t2

            t4 = time.perf_counter()
            preds = self.model(batch["lidar"], batch["map"])
            loss, metrics = self.criterion(preds, batch)
            _maybe_sync(self.device)
            t5 = time.perf_counter()
            t_forward += t5 - t4

            t6 = time.perf_counter()
            t_loss += t6 - t5

            t8 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            _maybe_sync(self.device)
            t9 = time.perf_counter()
            t_backward += t9 - t8

            t10 = time.perf_counter()
            self.optimizer.step()
            _maybe_sync(self.device)
            t11 = time.perf_counter()
            t_step += t11 - t10

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k].item()
            count += 1

        stats = {
            "loss": total_loss / max(count, 1),
            **{k: v / max(count, 1) for k, v in total_metrics.items()},
        }
        times = {
            "fetch_per_batch": t_fetch / max(count, 1),
            "to_device_per_batch": t_to_dev / max(count, 1),
            "forward_per_batch": t_forward / max(count, 1),
            "loss_per_batch": t_loss / max(count, 1),
            "backward_per_batch": t_backward / max(count, 1),
            "step_per_batch": t_step / max(count, 1),
            "batches": float(count),
        }
        return stats, times

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_metrics = {"rms_x": 0.0, "rms_y": 0.0, "pixel_error": 0.0}
        count = 0

        t_fetch = t_to_dev = t_forward = t_loss = 0.0

        it = iter(loader)
        num_batches = len(loader)

        for _ in range(num_batches):
            t0 = time.perf_counter()
            batch = next(it)
            t1 = time.perf_counter()
            t_fetch += t1 - t0

            t2 = time.perf_counter()
            batch = self._move_batch(batch)
            _maybe_sync(self.device)
            t3 = time.perf_counter()
            t_to_dev += t3 - t2

            t4 = time.perf_counter()
            preds = self.model(batch["lidar"], batch["map"])
            loss, metrics = self.criterion(preds, batch)
            _maybe_sync(self.device)
            t5 = time.perf_counter()
            t_forward += t5 - t4

            t6 = time.perf_counter()
            t_loss += t6 - t5

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k].item()
            count += 1

        stats = {
            "val_loss": total_loss / max(count, 1),
            "val_rms_x": total_metrics["rms_x"] / max(count, 1),
            "val_rms_y": total_metrics["rms_y"] / max(count, 1),
            "val_pixel_error": total_metrics["pixel_error"] / max(count, 1),
        }
        times = {
            "eval_fetch_per_batch": t_fetch / max(count, 1),
            "eval_to_device_per_batch": t_to_dev / max(count, 1),
            "eval_forward_per_batch": t_forward / max(count, 1),
            "eval_loss_per_batch": t_loss / max(count, 1),
            "eval_batches": float(count),
        }
        return stats, times


def create_history_dict() -> Dict[str, list]:
    return {
        "train_loss": [],
        "train_rms_x": [],
        "train_rms_y": [],
        "train_pixel_error": [],
        "val_loss": [],
        "val_rms_x": [],
        "val_rms_y": [],
        "val_pixel_error": [],
        "epoch_time": [],
    }


def build_optimizer(model: LocalizationModel, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )


def _ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_better(curr: float, best: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if mode == "min":
        return curr < best - min_delta
    if mode == "max":
        return curr > best + min_delta
    raise ValueError(f"Unknown mode: {mode}")


def _checkpoint_payload(
    *,
    epoch: int,
    model: LocalizationModel,
    optimizer: torch.optim.Optimizer,
    criterion: LocalizationCriterion,
    model_cfg: ModelConfig,
    optim_cfg: OptimConfig,
    save_cfg: SaveConfig,
    history: Dict[str, list],
    epoch_stats: Dict[str, float],
) -> Dict:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "criterion_state": criterion.state_dict(),
        "model_cfg": model_cfg,
        "optim_cfg": optim_cfg,
        "save_cfg": save_cfg,
        "history": history,
        "epoch_stats": epoch_stats,
    }


def save_checkpoint(payload: Dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resolve_checkpoint(
    save_dir: Union[str, Path],
    which: str = "best",
    filename_best: str = "best.ckpt",
    filename_last: str = "last.ckpt",
) -> Path:
    save_dir = Path(save_dir)
    filename = filename_best if which == "best" else filename_last
    return save_dir / filename


def load_localization_model(
    checkpoint_path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    return_optimizer: bool = True,
) -> Tuple[LocalizationModel, Optional[torch.optim.Optimizer], Dict]:
    try:
        from pathlib import WindowsPath
        add_safe_globals([DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig, WindowsPath])
    except ImportError:
        add_safe_globals([DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg: ModelConfig = checkpoint["model_cfg"]
    optim_cfg: OptimConfig = checkpoint["optim_cfg"]

    model = LocalizationModel(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device or optim_cfg.device)

    optimizer = None
    if return_optimizer:
        optimizer = build_optimizer(model, optim_cfg)
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, optimizer, checkpoint


class EarlyStopper:
    def __init__(self, cfg: EarlyStopConfig, best_ckpt_path: Optional[Union[str, Path]] = None):
        self.cfg = cfg
        self.best_value: Optional[float] = None
        self.wait = 0
        self.best_epoch = -1
        self.best_ckpt_path = Path(best_ckpt_path) if best_ckpt_path else None

    def step(self, epoch_idx: int, val_stats: Dict[str, float]) -> bool:
        if not self.cfg.enabled:
            return False
        monitor_value = val_stats.get(self.cfg.monitor)
        if monitor_value is None:
            return False
        mode = self.cfg.mode
        min_delta = self.cfg.min_delta
        improved = _is_better(monitor_value, self.best_value, mode, min_delta)
        if improved:
            self.best_value = monitor_value
            self.wait = 0
            self.best_epoch = epoch_idx
            if self.best_ckpt_path is not None:
                self.best_ckpt_path.touch(exist_ok=True)
        else:
            if epoch_idx >= self.cfg.warmup_epochs:
                self.wait += 1
            if self.wait > self.cfg.patience:
                return True
        return False


def train_localization_model(
    processed_raster_data_dir: Union[str, Path],
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    optim_cfg: OptimConfig,
    random_seed: int = 42,
    save_cfg: Optional[SaveConfig] = SaveConfig(),
    early_stop_cfg: Optional[EarlyStopConfig] = EarlyStopConfig(),
    subset_fraction: float = 1.0,
):
    root = Path(processed_raster_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise ValueError(f"No sample subfolders found in {root}")

    original_count = len(sample_dirs)
    if subset_fraction <= 0:
        raise ValueError("subset_fraction must be > 0")
    if subset_fraction < 1.0:
        rng = random.Random(random_seed)
        rng.shuffle(sample_dirs)
        subset_count = max(1, int(round(original_count * subset_fraction)))
        sample_dirs = sample_dirs[:subset_count]
        print(f"[Subset] Using {subset_count}/{original_count} samples (~{subset_fraction * 100:.1f}%).")

    dataset_cfg.sample_root = sample_dirs
    dataset_cfg.raster_builder = raster_builder_from_processed_dir

    full_dataset = GeoAlignRasterDataset(dataset_cfg)
    n_total = len(full_dataset)
    n_train = max(1, int(0.9 * n_total))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_loader = create_dataloader(train_ds, optim_cfg, shuffle=True)
    val_loader = create_dataloader(val_ds, optim_cfg, shuffle=False)

    batch = next(iter(train_loader))
    print("Sanity-check one batch shapes:")
    print("lidar:", batch["lidar"].shape)
    print("map  :", batch["map"].shape)

    model = LocalizationModel(model_cfg)
    criterion = LocalizationCriterion(model_cfg)
    optimizer = build_optimizer(model, optim_cfg)
    trainer = Trainer(model, criterion, optimizer, device=torch.device(optim_cfg.device))

    history = create_history_dict()

    best_value: Optional[float] = None
    best_ckpt_path: Optional[Path] = None
    if save_cfg and save_cfg.enable:
        save_dir = _ensure_dir(save_cfg.save_dir)
        best_ckpt_path = resolve_checkpoint(save_cfg.save_dir, "best", save_cfg.filename_best, save_cfg.filename_last)
        print(f"[Checkpointing] Enabled → saving to: {save_dir.resolve()}")
        print(f"[Checkpointing] Monitor: {save_cfg.monitor} | Mode: {save_cfg.mode}")
    else:
        print("[Checkpointing] Disabled")

    stopper = EarlyStopper(
        cfg=early_stop_cfg if early_stop_cfg is not None else EarlyStopConfig(enabled=False),
        best_ckpt_path=best_ckpt_path,
    )
    if stopper.cfg.enabled:
        print(
            "[EarlyStop] Enabled → monitor="
            f"{stopper.cfg.monitor}, mode={stopper.cfg.mode}, min_delta={stopper.cfg.min_delta}, "
            f"patience={stopper.cfg.patience}, warmup_epochs={stopper.cfg.warmup_epochs}, "
            f"restore_best={stopper.cfg.restore_best}",
        )
    else:
        print("[EarlyStop] Disabled")

    total_epochs = optim_cfg.epochs

    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()

        train_stats, train_times = trainer.train_epoch(train_loader)
        eval_start = time.perf_counter()
        val_stats, eval_times = trainer.evaluate(val_loader)
        eval_time_total = time.perf_counter() - eval_start

        history["train_loss"].append(train_stats["loss"])
        history["train_rms_x"].append(train_stats["rms_x"])
        history["train_rms_y"].append(train_stats["rms_y"])
        history["train_pixel_error"].append(train_stats["pixel_error"])
        history["val_loss"].append(val_stats["val_loss"])
        history["val_rms_x"].append(val_stats["val_rms_x"])
        history["val_rms_y"].append(val_stats["val_rms_y"])
        history["val_pixel_error"].append(val_stats["val_pixel_error"])

        epoch_time = time.perf_counter() - epoch_start
        history["epoch_time"].append(epoch_time)
        epochs_done = epoch + 1

        avg_epoch_time = sum(history["epoch_time"]) / len(history["epoch_time"])
        remaining_epochs = total_epochs - epochs_done
        est_remaining_secs = avg_epoch_time * remaining_epochs
        finish_at_dt = datetime.now() + timedelta(seconds=est_remaining_secs)

        ckpt_time = 0.0
        if save_cfg and save_cfg.enable:
            ckpt_t0 = time.perf_counter()
            payload = _checkpoint_payload(
                epoch=epochs_done,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                model_cfg=model_cfg,
                optim_cfg=optim_cfg,
                save_cfg=save_cfg,
                history=history,
                epoch_stats={**train_stats, **val_stats},
            )
            if save_cfg.save_last:
                save_checkpoint(
                    payload,
                    resolve_checkpoint(save_cfg.save_dir, "last", save_cfg.filename_best, save_cfg.filename_last),
                )

            if save_cfg.save_best and save_cfg.monitor in val_stats:
                current_value = float(val_stats[save_cfg.monitor])
                if _is_better(current_value, best_value, save_cfg.mode):
                    best_value = current_value
                    save_checkpoint(
                        payload,
                        resolve_checkpoint(save_cfg.save_dir, "best", save_cfg.filename_best, save_cfg.filename_last),
                    )
                    print(f"[Checkpointing] New best {save_cfg.monitor}={current_value:.6f} → saved BEST at epoch {epochs_done}")
            elif save_cfg.save_best and save_cfg.monitor not in val_stats:
                print(f"[Checkpointing] WARNING: monitor '{save_cfg.monitor}' not in val_stats; skipping best-save.")
            ckpt_time = time.perf_counter() - ckpt_t0

        def fmt(ms: float) -> str:
            return f"{ms * 1000:.1f} ms"

        print(
            f"Epoch {epochs_done:02d}/{total_epochs}: "
            f"train_loss={train_stats['loss']:.4f}  "
            f"| train_rms_x={train_stats['rms_x']:.3f} m  "
            f"| train_rms_y={train_stats['rms_y']:.3f} m  "
            f"| train_pixel_error={train_stats['pixel_error']:.3f} px  ||  "
            f"val_loss={val_stats['val_loss']:.4f}  "
            f"| val_rms_x={val_stats['val_rms_x']:.3f} m  "
            f"| val_rms_y={val_stats['val_rms_y']:.3f} m  "
            f"| val_pixel_error={val_stats['val_pixel_error']:.3f} px",
        )
        print(
            f"  ⏲️ Epoch wall time={epoch_time:.2f}s  "
            f"| eval total={eval_time_total:.2f}s  "
            f"| ckpt={ckpt_time:.2f}s  "
            f"| remaining≈{_fmt_secs(est_remaining_secs)}  "
            f"(ETA ~ {finish_at_dt.strftime('%Y-%m-%d %H:%M:%S')})",
        )

        if stopper.step(epochs_done, val_stats):
            print(
                f"[EarlyStop] Triggered at epoch {epochs_done} "
                f"(no improvement in '{stopper.cfg.monitor}' for > {stopper.cfg.patience} epochs).",
            )
            if (
                stopper.cfg.restore_best
                and save_cfg
                and save_cfg.enable
                and best_ckpt_path
                and best_ckpt_path.exists()
            ):
                print(f"[EarlyStop] Restoring best weights from: {best_ckpt_path}")
                best_model, _, _ = load_localization_model(best_ckpt_path, device=optim_cfg.device, return_optimizer=False)
                model.load_state_dict(best_model.state_dict())
            break

    print("Training complete.")
    return model, criterion, optimizer, trainer, history


def _fmt_secs(secs: float) -> str:
    secs = int(max(secs, 0))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


__all__ = [
    "Trainer",
    "EarlyStopper",
    "build_optimizer",
    "save_checkpoint",
    "resolve_checkpoint",
    "load_localization_model",
    "train_localization_model",
]
