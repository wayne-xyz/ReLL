from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig
from .gaussian_peak_refine import gaussian_peak_refine
from .theta_peak_refine import theta_peak_refine_batch


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
            # Keep original resolution: stride=1 for all depths (no downsampling)
            stride = 1
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
        # No downsampling: factor = 1 (original resolution preserved)
        self.downsampling_factor = 1

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

        # Store downsampling factor from encoder (both encoders should have the same factor)
        self.downsampling_factor = self.lidar_encoder.downsampling_factor

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

        # Store theta search config (not as buffer - will be created at runtime)
        self.theta_search_deg = max(int(round(getattr(config, "theta_search_deg", 0))), 0)

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
        patches = geo_padded.unfold(2, H, 1).unfold(3, W, 1)
        cost = torch.einsum("bchw,bcijhw->bij", online, patches)
        # cost = self.translation_smoother(cost.unsqueeze(1)).squeeze(1)
        return cost

    def compute_theta_logits(self, online: Tensor, geo: Tensor, theta_candidates_deg: Tensor) -> Tensor:
        """
        Compute rotation similarity scores for given theta candidates.

        Args:
            online: LiDAR embeddings [B, C, H, W]
            geo: Map embeddings [B, C, H, W]
            theta_candidates_deg: Rotation angles to search in degrees [num_angles]

        Returns:
            Rotation logits [B, num_angles]
        """
        angles_deg = theta_candidates_deg.to(online.device, dtype=online.dtype)
        if angles_deg.numel() == 1 and torch.allclose(angles_deg, torch.zeros_like(angles_deg)):
            return (online * geo).sum(dim=(1, 2, 3), keepdim=True)

        B = online.shape[0]
        logits = []
        for angle_deg in angles_deg:
            angle_rad = angle_deg * math.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            theta = torch.zeros(B, 2, 3, device=online.device, dtype=online.dtype)
            theta[:, 0, 0] = cos_a
            theta[:, 0, 1] = -sin_a
            theta[:, 1, 0] = sin_a
            theta[:, 1, 1] = cos_a
            grid = F.affine_grid(theta, size=geo.size(), align_corners=False)
            rotated = F.grid_sample(geo, grid, align_corners=False, mode="bilinear", padding_mode="border")
            sim = (online * rotated).sum(dim=(1, 2, 3))
            logits.append(sim)
        return torch.stack(logits, dim=1)

    def forward(
        self,
        lidar: Tensor,
        geospatial: Tensor,
        theta_candidates_deg: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for localization.

        Args:
            lidar: LiDAR input [B, C_lidar, H, W]
            geospatial: Map input [B, C_map, H, W]
            theta_candidates_deg: Optional rotation angles to search [num_angles].
                                 If None, creates default from config.

        Returns:
            Dictionary with translation_logits, theta_logits, embeddings
        """
        lidar_feat = self._encode(lidar, self.lidar_adapter, self.lidar_encoder)
        map_feat = self._encode(geospatial, self.map_adapter, self.map_encoder)

        lidar_proj = self._l2norm(self.lidar_projection(lidar_feat))
        map_proj = self._l2norm(self.map_projection(map_feat))

        translation_logits = self.compute_translation_logits(lidar_proj, map_proj)

        # Create default theta candidates if not provided
        if theta_candidates_deg is None:
            theta_range = self.theta_search_deg
            if theta_range > 0:
                # Use 0.5° steps for smoother visualization and better precision
                theta_candidates_deg = torch.arange(
                    -theta_range, theta_range + 0.5, step=0.5,
                    dtype=lidar_proj.dtype, device=lidar_proj.device
                )
            else:
                theta_candidates_deg = torch.zeros(1, dtype=lidar_proj.dtype, device=lidar_proj.device)

        theta_logits = self.compute_theta_logits(lidar_proj, map_proj, theta_candidates_deg)

        return {
            "translation_logits": translation_logits,
            "theta_logits": theta_logits,
            "theta_candidates_deg": theta_candidates_deg,  # Return candidates used
            "lidar_embedding": lidar_proj,
            "map_embedding": map_proj,
        }


class LocalizationCriterion(nn.Module):
    def __init__(self, model_config: ModelConfig, downsampling_factor: int = 2) -> None:
        super().__init__()
        r = model_config.search_radius
        coords = torch.linspace(-r, r, steps=2 * r + 1)
        self.register_buffer("coord_grid_x", coords.view(1, 1, -1).repeat(1, 2 * r + 1, 1))
        self.register_buffer("coord_grid_y", coords.view(1, -1, 1).repeat(1, 1, 2 * r + 1))
        self.search_radius = r
        self.sigma_px = max(float(model_config.gaussian_sigma_px), 1e-6)
        self.w_xy = float(model_config.w_xy)
        self.w_theta = float(model_config.w_theta)
        self.sigma_weight_xy = float(getattr(model_config, "sigma_weight_xy", 1.0))
        self.sigma_weight_theta = float(getattr(model_config, "sigma_weight_theta", 1.0))
        # Store theta config (not as buffer - created at runtime)
        self.theta_search_deg = max(int(round(getattr(model_config, "theta_search_deg", 0))), 0)
        self.sigma_theta_deg = max(float(getattr(model_config, "gaussian_sigma_theta_deg", 1.0)), 1e-6)
        # Store downsampling factor (feature space is downsampled by this factor)
        self.downsampling_factor = downsampling_factor

    def _gaussian_peak_refine(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply advanced Gaussian peak refinement using the sophisticated method from gaussian_peak_refine.py.

        This method combines multiple refinement strategies:
        - Weighted centroid baseline
        - One-dimensional quadratic refinement
        - Newton-Raphson iterative optimization
        - Softmax expectation
        - 2D quadratic fitting
        - Blending with adaptive penalties

        Args:
            logits: (B, H, W) correlation scores from sliding window

        Returns:
            mu_x_px: (B,) refined x position in pixels
            mu_y_px: (B,) refined y position in pixels
            scores: (B,) refined peak scores
        """
        B, H, W = logits.shape
        device = logits.device

        mu_x_refined = []
        mu_y_refined = []
        scores_refined = []

        for b in range(B):
            heatmap = logits[b]  # (H, W)

            # Apply the advanced Gaussian peak refinement
            result = gaussian_peak_refine(heatmap)

            mu_x_refined.append(result.x)
            mu_y_refined.append(result.y)
            scores_refined.append(result.score)

        mu_x_px = torch.tensor(mu_x_refined, device=device, dtype=logits.dtype)
        mu_y_px = torch.tensor(mu_y_refined, device=device, dtype=logits.dtype)
        scores = torch.tensor(scores_refined, device=device, dtype=logits.dtype)

        return mu_x_px, mu_y_px, scores

    def forward(self, predictions: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        logits = predictions["translation_logits"]
        device = logits.device
        B, H, W = logits.shape

        # Apply downsampling factor to resolution (feature space is downsampled)
        resolution = batch["resolution"].to(device).view(-1) * self.downsampling_factor
        pose_mu = batch["pose_mu"].to(device)

        coord_x = self.coord_grid_x.to(device).expand(B, -1, -1)
        coord_y = self.coord_grid_y.to(device).expand(B, -1, -1)

        mu_gt_x = pose_mu[:, 0]
        mu_gt_y = pose_mu[:, 1]

        # ===== Level 1: Pixel-level position (discrete argmax) =====
        flat_logits = logits.view(B, -1)
        argmax_indices = torch.argmax(flat_logits, dim=-1)
        argmax_row = argmax_indices // W
        argmax_col = argmax_indices % W
        pxlevel_x_px = coord_x[torch.arange(B), argmax_row, argmax_col]
        pxlevel_y_px = coord_y[torch.arange(B), argmax_row, argmax_col]
        pxlevel_x_m = pxlevel_x_px * resolution
        pxlevel_y_m = pxlevel_y_px * resolution

        # Calculate pixel-level RMS (for monitoring only)
        pxlevel_err_x = pxlevel_x_m - mu_gt_x
        pxlevel_err_y = pxlevel_y_m - mu_gt_y
        pxlevel_rms_x = torch.sqrt(torch.mean(pxlevel_err_x ** 2) + 1e-9)
        pxlevel_rms_y = torch.sqrt(torch.mean(pxlevel_err_y ** 2) + 1e-9)

        # ===== Level 2: Advanced Gaussian peak fitting (for MONITORING only - not differentiable!) =====
        with torch.no_grad():  # Gaussian fitting doesn't preserve gradients
            # Advanced Gaussian refinement (multi-strategy blended approach from gaussian_peak_refine.py)
            gaussian_x_px, gaussian_y_px, gaussian_scores = self._gaussian_peak_refine(logits)
            gaussian_x_m = gaussian_x_px * resolution
            gaussian_y_m = gaussian_y_px * resolution

            # Calculate Gaussian-fitted RMS (for monitoring)
            gaussian_err_x = gaussian_x_m - mu_gt_x
            gaussian_err_y = gaussian_y_m - mu_gt_y
            gaussian_rms_x = torch.sqrt(torch.mean(gaussian_err_x ** 2) + 1e-9)
            gaussian_rms_y = torch.sqrt(torch.mean(gaussian_err_y ** 2) + 1e-9)

        # ===== LOSS: Use differentiable softmax expectation (NOT Gaussian!) =====
        # Softmax provides sub-pixel accuracy AND preserves gradients for learning
        prob = F.softmax(logits.view(B, -1), dim=-1).view(B, H, W)

        # Compute expected position (differentiable weighted average)
        mu_hat_x_px = (prob * coord_x).sum(dim=(1, 2))
        mu_hat_y_px = (prob * coord_y).sum(dim=(1, 2))
        mu_hat_x_m = mu_hat_x_px * resolution
        mu_hat_y_m = mu_hat_y_px * resolution

        # Compute variance (uncertainty estimation)
        dx_px = coord_x - mu_hat_x_px.view(-1, 1, 1)
        dy_px = coord_y - mu_hat_y_px.view(-1, 1, 1)
        var_x_px = (prob * (dx_px ** 2)).sum(dim=(1, 2))
        var_y_px = (prob * (dy_px ** 2)).sum(dim=(1, 2))
        sigma_hat_x_m = torch.sqrt(var_x_px.clamp_min(1e-9)) * resolution
        sigma_hat_y_m = torch.sqrt(var_y_px.clamp_min(1e-9)) * resolution
        sigma_target_m = resolution * self.sigma_px

        # Compute loss (differentiable!)
        loss_mu_xy = torch.abs(mu_hat_x_m - mu_gt_x) + torch.abs(mu_hat_y_m - mu_gt_y)
        loss_sigma_xy = torch.abs(sigma_hat_x_m - sigma_target_m) + torch.abs(sigma_hat_y_m - sigma_target_m)
        loss_xy = (loss_mu_xy + self.sigma_weight_xy * loss_sigma_xy).mean()

        # Calculate softmax-refined RMS (this is what the model learns to minimize)
        softmax_err_x = mu_hat_x_m - mu_gt_x
        softmax_err_y = mu_hat_y_m - mu_gt_y
        softmax_rms_x = torch.sqrt(torch.mean(softmax_err_x ** 2) + 1e-9)
        softmax_rms_y = torch.sqrt(torch.mean(softmax_err_y ** 2) + 1e-9)

        sigma_err_x = torch.abs(sigma_hat_x_m - sigma_target_m).mean()
        sigma_err_y = torch.abs(sigma_hat_y_m - sigma_target_m).mean()

        if self.theta_search_deg > 0:
            theta_logits = predictions["theta_logits"].to(device)
            theta_candidates_deg = predictions["theta_candidates_deg"].to(device)
            mu_theta_gt = pose_mu[:, 2]

            # Use the reusable theta peak refinement module
            mu_hat_theta_deg, sigma_hat_theta_deg = theta_peak_refine_batch(theta_logits, theta_candidates_deg)
            sigma_theta_target = theta_logits.new_full((B,), self.sigma_theta_deg)

            loss_theta_mu = torch.abs(mu_hat_theta_deg - mu_theta_gt)
            loss_theta_sigma = torch.abs(sigma_hat_theta_deg - sigma_theta_target)
            loss_theta = (loss_theta_mu + self.sigma_weight_theta * loss_theta_sigma).mean()

            err_theta = mu_hat_theta_deg - mu_theta_gt
            rms_theta = torch.sqrt(torch.mean(err_theta ** 2) + 1e-9)
            sigma_err_theta = torch.abs(sigma_hat_theta_deg - sigma_theta_target).mean()
        else:
            loss_theta = logits.new_tensor(0.0)
            rms_theta = logits.new_tensor(0.0)
            sigma_err_theta = logits.new_tensor(0.0)

        loss = self.w_xy * loss_xy + self.w_theta * loss_theta

        # Three levels for comparison:
        # 1. Pixel-level: Discrete argmax from sliding window
        # 2. Softmax: Differentiable sub-pixel (USED IN LOSS - model learns this)
        # 3. Gaussian: Advanced multi-strategy blended refinement (monitoring only)
        metrics = {
            "pxlevel_rms_x": pxlevel_rms_x,        # Level 1: Discrete
            "pxlevel_rms_y": pxlevel_rms_y,
            "softmax_rms_x": softmax_rms_x,        # Level 2: Softmax (LOSS target)
            "softmax_rms_y": softmax_rms_y,
            "gaussian_rms_x": gaussian_rms_x,      # Level 3: Advanced Gaussian (monitoring)
            "gaussian_rms_y": gaussian_rms_y,
            "rms_theta": rms_theta,
            "sigma_err_x": sigma_err_x,
            "sigma_err_y": sigma_err_y,
            "sigma_err_theta": sigma_err_theta,
        }
        return loss, metrics


__all__ = ["LocalizationModel", "LocalizationCriterion"]
