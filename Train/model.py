from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, depth: int, base_channels: int) -> None:
        super().__init__()
        stem_channels = max(base_channels // 2, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(stem_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_channels_iter = stem_channels
        for i in range(depth):
            out_channels = min(embed_dim, base_channels * (2 ** i))
            stride = 2 if i < depth - 1 else 1
            block = nn.Sequential(
                nn.Conv2d(in_channels_iter, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
            layers.append(block)
            in_channels_iter = out_channels
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_channels_iter, embed_dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class LocalizationModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.lidar_adapter = nn.Sequential(
            nn.Conv2d(config.lidar_in_channels, config.stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(config.stem_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.map_adapter = nn.Sequential(
            nn.Conv2d(config.map_in_channels, config.stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(config.stem_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.lidar_encoder = PyramidEncoder(
            in_channels=config.stem_channels,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            base_channels=config.encoder_base_channels,
        )
        self.map_encoder = PyramidEncoder(
            in_channels=config.stem_channels,
            embed_dim=config.embed_dim,
            depth=config.encoder_depth,
            base_channels=config.encoder_base_channels,
        )

        if config.translation_smoothing_kernel < 1 or config.translation_smoothing_kernel % 2 == 0:
            raise ValueError("translation_smoothing_kernel must be a positive odd integer.")
        self.lidar_projection = nn.Conv2d(config.embed_dim, config.proj_dim, kernel_size=1, bias=False)
        self.map_projection = nn.Conv2d(config.embed_dim, config.proj_dim, kernel_size=1, bias=False)
        self.translation_smoother = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=config.translation_smoothing_kernel,
            padding=config.translation_smoothing_kernel // 2,
            bias=False,
        )
        nn.init.constant_(self.translation_smoother.weight, 1.0 / (config.translation_smoothing_kernel ** 2))
        self.translation_smoother.weight.requires_grad_(False)

    def _encode(self, x: Tensor, adapter: nn.Module, encoder: nn.Module) -> Tensor:
        x = adapter(x)
        x = encoder(x)
        return x

    def _l2norm(self, emb: Tensor) -> Tensor:
        return torch.nan_to_num(F.normalize(emb, p=2, dim=1, eps=1e-6))

    def compute_translation_logits(self, online: Tensor, geo: Tensor) -> Tensor:
        radius = self.config.search_radius
        geo_padded = F.pad(geo, pad=(radius, radius, radius, radius), mode="replicate")
        B, C, H, W = online.shape

        patches = geo_padded.unfold(2, H, 1).unfold(3, W, 1)  # B, C, 2r+1, 2r+1, H, W
        cost = torch.einsum("bchw,bcijhw->bij", online, patches)
        cost = self.translation_smoother(cost.unsqueeze(1)).squeeze(1)
        return cost

    def forward(self, lidar: Tensor, geospatial: Tensor) -> Dict[str, Tensor]:
        lidar_feat = self._encode(lidar, self.lidar_adapter, self.lidar_encoder)
        map_feat = self._encode(geospatial, self.map_adapter, self.map_encoder)

        lidar_proj = self._l2norm(self.lidar_projection(lidar_feat))
        map_proj = self._l2norm(self.map_projection(map_feat))

        translation_logits = self.compute_translation_logits(lidar_proj, map_proj)

        return {
            "translation_logits": translation_logits,
            "lidar_embedding": lidar_proj,
            "map_embedding": map_proj,
        }


class LocalizationCriterion(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        r = model_config.search_radius
        coords = torch.linspace(-r, r, steps=2 * r + 1)
        self.register_buffer("coord_grid_x", coords.view(1, 1, -1).repeat(1, 2 * r + 1, 1))
        self.register_buffer("coord_grid_y", coords.view(1, -1, 1).repeat(1, 1, 2 * r + 1))
        self.search_radius = r
        self.sigma_px = max(float(model_config.gaussian_sigma_px), 1e-6)
        self.w_xy = float(model_config.w_xy)

    def forward(self, predictions: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits = predictions["translation_logits"]
        device = logits.device
        B, H, W = logits.shape

        resolution = batch["resolution"].to(device)
        pose_mu = batch["pose_mu"].to(device)

        mu_px_x = pose_mu[:, 0] / resolution
        mu_px_y = pose_mu[:, 1] / resolution

        coord_x = self.coord_grid_x.to(device).expand(B, -1, -1)
        coord_y = self.coord_grid_y.to(device).expand(B, -1, -1)

        dx = coord_x - mu_px_x.view(-1, 1, 1)
        dy = coord_y - mu_px_y.view(-1, 1, 1)
        gaussian = torch.exp(-0.5 * (dx ** 2 + dy ** 2) / (self.sigma_px ** 2 + 1e-9))
        gaussian = gaussian / gaussian.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)

        flat_logits = logits.view(B, -1)
        log_prob = F.log_softmax(flat_logits, dim=-1)
        loss_xy = -(gaussian.view(B, -1) * log_prob).sum(dim=-1).mean()

        prob = log_prob.exp().view(B, H, W)
        pred_idx = flat_logits.argmax(dim=-1)
        width = 2 * self.search_radius + 1
        pred_y = pred_idx // width
        pred_x = pred_idx % width
        pred_dx_px = pred_x.float() - self.search_radius
        pred_dy_px = pred_y.float() - self.search_radius

        pred_dx_m = pred_dx_px * resolution
        pred_dy_m = pred_dy_px * resolution

        err_x = pred_dx_m - pose_mu[:, 0]
        err_y = pred_dy_m - pose_mu[:, 1]
        pixel_err = torch.sqrt((err_x / (resolution + 1e-9)) ** 2 + (err_y / (resolution + 1e-9)) ** 2)

        rms_x = torch.sqrt(torch.mean(err_x ** 2) + 1e-9)
        rms_y = torch.sqrt(torch.mean(err_y ** 2) + 1e-9)
        err_theta = -pose_mu[:, 2]
        rms_theta = torch.sqrt(torch.mean(err_theta ** 2) + 1e-9)

        loss = self.w_xy * loss_xy

        metrics = {
            "rms_x": rms_x,
            "rms_y": rms_y,
            "rms_theta": rms_theta,
            "pixel_error": pixel_err.mean(),
        }
        return loss, metrics


__all__ = ["LocalizationModel", "LocalizationCriterion"]
