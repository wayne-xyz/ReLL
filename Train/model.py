from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import ModelConfig


def build_theta_grid(theta_bins: int, theta_range_deg: float) -> Tensor:
    assert theta_bins >= 3, "theta_bins must be >= 3"
    return torch.linspace(-theta_range_deg, theta_range_deg, theta_bins)


def _make_conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.1, inplace=True),
    )


class HeightAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(4, in_channels // 2)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=0.1)
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        ctx = x.mean(dim=(2, 3), keepdim=True)
        w = self.mlp(ctx)
        return x * w


class HeightAwareEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, depth: int, base_channels: int = 64, use_height_attn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_channels

        out_channels = min(embed_dim, max(base_channels * (2 ** 0), 32))
        layers.append(_make_conv_block(channels, out_channels, stride=1))
        channels = out_channels

        self.attn = HeightAttention(out_channels) if use_height_attn else None

        for i in range(1, depth):
            out_channels = min(embed_dim, max(base_channels * (2 ** i), 32))
            layers.append(_make_conv_block(channels, out_channels, stride=1))
            channels = out_channels

        layers.append(_make_conv_block(channels, embed_dim, stride=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder[0](x)
        if self.attn is not None:
            x = self.attn(x)
        for layer in self.encoder[1:]:
            x = layer(x)
        return x


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, depth: int, base_channels: int = 64):
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_channels
        for i in range(depth):
            out_channels = min(embed_dim, max(base_channels * (2 ** i), 32))
            layers.append(_make_conv_block(channels, out_channels, stride=1))
            channels = out_channels
        layers.append(_make_conv_block(channels, embed_dim, stride=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class LocalizationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.online_encoder = HeightAwareEncoder(
            config.lidar_in_channels,
            config.embed_dim,
            config.encoder_depth,
            use_height_attn=config.height_attention,
        )
        self.geo_encoder = PyramidEncoder(
            config.map_in_channels,
            config.embed_dim,
            config.encoder_depth,
        )

        self.proj_online = nn.Conv2d(config.embed_dim, config.proj_dim, kernel_size=1, bias=False)
        self.proj_geo = nn.Conv2d(config.embed_dim, config.proj_dim, kernel_size=1, bias=False)

        self.theta_grid = build_theta_grid(config.theta_bins, config.theta_range_deg)

    def _l2norm(self, emb: Tensor) -> Tensor:
        return torch.nan_to_num(F.normalize(emb, p=2, dim=1, eps=1e-6))

    def compute_translation_cost(self, online: Tensor, geo: Tensor) -> Tensor:
        radius = self.config.search_radius
        pad = (radius, radius, radius, radius)
        geo_padded = F.pad(geo, pad=pad, mode="constant", value=0.0)

        B, C, H, W = online.shape
        _, _, H_p, W_p = geo_padded.shape

        geo_inp = geo_padded.reshape(1, B * C, H_p, W_p)
        kernel_wt = online
        cost = F.conv2d(geo_inp, kernel_wt, padding=0, groups=B)
        return cost.squeeze(0)

    def compute_orientation_cost(self, online: Tensor, geo: Tensor) -> Tensor:
        device = online.device
        theta_grid_rad = self.theta_grid.to(device) * (math.pi / 180.0)

        B, C, H, W = online.shape
        K = theta_grid_rad.shape[0]
        cos_t = torch.cos(theta_grid_rad)
        sin_t = torch.sin(theta_grid_rad)

        matrices = torch.zeros((B, K, 2, 3), device=device, dtype=online.dtype)
        matrices[:, :, 0, 0] = cos_t.view(1, K)
        matrices[:, :, 0, 1] = -sin_t.view(1, K)
        matrices[:, :, 1, 0] = sin_t.view(1, K)
        matrices[:, :, 1, 1] = cos_t.view(1, K)

        matrices = matrices.view(B * K, 2, 3)
        grid = F.affine_grid(matrices, [B * K, C, H, W], align_corners=False)

        online_rep = online.repeat_interleave(K, dim=0)
        geo_rep = geo.repeat_interleave(K, dim=0)

        rotated = F.grid_sample(online_rep, grid, align_corners=False, mode="bilinear")
        scores = (rotated * geo_rep).sum(dim=(1, 2, 3))
        return scores.view(B, K)

    def forward(self, lidar: Tensor, geospatial: Tensor) -> Dict[str, Tensor]:
        lidar_feat_raw = self.online_encoder(lidar)
        geo_feat_raw = self.geo_encoder(geospatial)

        lidar_feat = self._l2norm(self.proj_online(lidar_feat_raw))
        geo_feat = self._l2norm(self.proj_geo(geo_feat_raw))
        lidar_feat = torch.nan_to_num(lidar_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        geo_feat = torch.nan_to_num(geo_feat, nan=0.0, posinf=1e6, neginf=-1e6)

        translation_cost = self.compute_translation_cost(lidar_feat, geo_feat)
        orientation_cost = self.compute_orientation_cost(lidar_feat, geo_feat)

        return {
            "online_embedding": lidar_feat,
            "geospatial_embedding": geo_feat,
            "translation_cost": translation_cost,
            "orientation_cost": orientation_cost,
        }


def three_point_log_parabola_subpixel(p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    eps = 1e-12
    y1 = (p1 + eps).log()
    y2 = (p2 + eps).log()
    y3 = (p3 + eps).log()
    denom = (y1 - 2.0 * y2 + y3).clamp(min=1e-6)
    delta = 0.5 * (y1 - y3) / denom
    return delta.clamp(-1.0, 1.0)


def gaussian_subpixel_refine_2d_scores(
    scores: Tensor,
    eps: float = 1e-6,
    return_gaussian_params: bool = False,
    vis: bool = False,
    vis_n: int = 2,
    vis_res: int = 41,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    assert scores.ndim == 3, "scores must be [B,H,W]"
    scores = scores.float()
    B, H, W = scores.shape

    flat = scores.view(B, -1)
    idx = flat.argmax(dim=-1)
    y0 = (idx // W).to(torch.int64)
    x0 = (idx % W).to(torch.int64)

    x = x0.clamp(1, W - 2)
    y = y0.clamp(1, H - 2)

    xs = torch.tensor([-1.0, 0.0, 1.0], device=scores.device)
    ys = torch.tensor([-1.0, 0.0, 1.0], device=scores.device)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)

    phi = torch.stack([xx**2, yy**2, xx * yy, xx, yy, torch.ones_like(xx)], dim=-1)
    Phi = phi.unsqueeze(0).expand(B, -1, -1)

    patches = torch.stack(
        [
            scores[torch.arange(B), (y + dy).clamp(0, H - 1), (x + dx).clamp(0, W - 1)]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ],
        dim=-1,
    )

    minv = patches.min(dim=-1, keepdim=True).values
    y_pos = patches - minv + eps
    y_log = torch.log(y_pos).unsqueeze(-1)

    lam = 1e-6
    Pt = Phi.transpose(1, 2)
    PtP = torch.matmul(Pt, Phi)
    reg = lam * torch.eye(6, device=scores.device).unsqueeze(0)
    PtY = torch.matmul(Pt, y_log)
    theta = torch.linalg.solve(PtP + reg, PtY).squeeze(-1)

    a, b, c, d, e, f0 = [theta[:, i] for i in range(6)]

    Hmat = torch.stack(
        [
            torch.stack([2 * a, c], dim=-1),
            torch.stack([c, 2 * b], dim=-1),
        ],
        dim=-2,
    )
    rhs = -torch.stack([d, e], dim=-1).unsqueeze(-1)

    xy = torch.linalg.solve(Hmat, rhs).squeeze(-1)
    dx = xy[:, 0].clamp(-1.0, 1.0)
    dy = xy[:, 1].clamp(-1.0, 1.0)

    x_ref = x.float() + dx
    y_ref = y.float() + dy

    A = None
    Sigma = None
    if return_gaussian_params:
        damp = 1e-9
        I2 = torch.eye(2, device=scores.device).unsqueeze(0).expand(B, -1, -1)
        Sigma = -torch.linalg.inv(Hmat + damp * I2)
        logA = (a * dx * dx + b * dy * dy + c * dx * dy + d * dx + e * dy + f0)
        A = torch.exp(logA)

    return x_ref, y_ref, A, Sigma


class LocalizationCriterion(nn.Module):
    """
    Sub-pixel via local fits:
      - 2D translation: 2-D log-Gaussian fit around the peak → mean (x,y) and Σ.
      - 1D yaw: 3-point log-parabola around peak → mean θ and std.

    Loss = w_xy*L1(x,y) + w_theta*L1(theta) + alpha_xy*L1(sigma_x,sigma_y) + alpha_theta*L1(sigma_theta)
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        r = model_config.search_radius
        coords = torch.linspace(-r, r, steps=2 * r + 1)
        self.register_buffer("coord_grid_x", coords.view(1, 1, -1).repeat(1, 2 * r + 1, 1))
        self.register_buffer("coord_grid_y", coords.view(1, -1, 1).repeat(1, 1, 2 * r + 1))

        theta_coords_deg = torch.linspace(
            -model_config.theta_range_deg,
            model_config.theta_range_deg,
            steps=model_config.theta_bins,
        )
        self.register_buffer("theta_coords", theta_coords_deg * math.pi / 180.0)

        self.w_xy = float(model_config.w_xy)
        self.w_theta = float(model_config.w_theta)
        self.alpha_xy = float(model_config.alpha_xy)
        self.alpha_theta = float(model_config.alpha_theta)

    @staticmethod
    def _wrap_angle(diff: Tensor) -> Tensor:
        return (diff + math.pi).remainder(2 * math.pi) - math.pi

    def _subpixel_xy_from_scores(self, T: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        T = T.float()
        T = T - T.amax(dim=(-1, -2), keepdim=True)
        x_ref, y_ref, _, Sigma = gaussian_subpixel_refine_2d_scores(
            T,
            return_gaussian_params=True,
        )
        if Sigma is None:
            B, _, _ = T.shape
            Sigma = torch.full((B, 2, 2), 1e-4, device=T.device, dtype=T.dtype)
        return x_ref, y_ref, Sigma

    def _yaw_mean_std_from_scores(self, O: Tensor) -> Tuple[Tensor, Tensor]:
        O = O.float()
        B, K = O.shape
        k0 = O.argmax(dim=-1)
        km1 = (k0 - 1) % K
        kp1 = (k0 + 1) % K
        b = torch.arange(B, device=O.device)

        Omax = O.max(dim=-1, keepdim=True).values
        P = torch.exp(O - Omax)

        p1, p2, p3 = P[b, km1], P[b, k0], P[b, kp1]
        y1, y2, y3 = (p1 + 1e-12).log(), (p2 + 1e-12).log(), (p3 + 1e-12).log()
        a = 0.5 * (y1 + y3 - 2.0 * y2)
        delta = three_point_log_parabola_subpixel(p1, p2, p3)

        sigma_bins = torch.sqrt(torch.clamp(-1.0 / (2.0 * a + 1e-12), min=1e-9))

        theta_k = self.theta_coords
        step = (theta_k[1] - theta_k[0]).item()
        theta0 = theta_k[k0]
        theta_ref = theta0 + delta * step
        theta_std = sigma_bins * abs(step)
        return theta_ref, theta_std

    def forward(self, predictions: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        T = predictions["translation_cost"].float()
        O = predictions["orientation_cost"].float()
        B, H, W = T.shape

        res = batch["resolution"].to(T.device)
        mu_gt = batch["pose_mu"].to(T.device)
        sigma_gt = batch["pose_sigma"].to(T.device)

        x_ref, y_ref, Sigma = self._subpixel_xy_from_scores(T)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        dx_px = x_ref - cx
        dy_px = y_ref - cy
        dx_m = dx_px * res
        dy_m = dy_px * res

        var_px_x = torch.clamp(Sigma[:, 0, 0], min=1e-12)
        var_px_y = torch.clamp(Sigma[:, 1, 1], min=1e-12)
        sigma_m_x = torch.sqrt(var_px_x) * res
        sigma_m_y = torch.sqrt(var_px_y) * res

        theta_ref, theta_std = self._yaw_mean_std_from_scores(O)

        L_xy = (dx_m - mu_gt[:, 0]).abs() + (dy_m - mu_gt[:, 1]).abs()
        angdiff = self._wrap_angle(theta_ref - mu_gt[:, 2])
        L_th = angdiff.abs()

        L_sig_xy = (sigma_m_x - sigma_gt[:, 0]).abs() + (sigma_m_y - sigma_gt[:, 1]).abs()
        L_sig_th = (theta_std - sigma_gt[:, 2]).abs()

        loss = (
            self.w_xy * L_xy.mean()
            + self.w_theta * L_th.mean()
            + self.alpha_xy * L_sig_xy.mean()
            + self.alpha_theta * L_sig_th.mean()
        )

        err_x = dx_m - mu_gt[:, 0]
        err_y = dy_m - mu_gt[:, 1]
        err_theta = angdiff

        metrics = {
            "mean_abs_xy": err_x.abs().mean() + err_y.abs().mean(),
            "mean_abs_theta": L_th.mean(),
            "sigma_xy_mae": L_sig_xy.mean(),
            "sigma_theta_mae": L_sig_th.mean(),
            "rms_x": torch.sqrt(torch.mean(err_x ** 2)),
            "rms_y": torch.sqrt(torch.mean(err_y ** 2)),
            "rms_theta": torch.sqrt(torch.mean(err_theta ** 2)),
        }
        return loss, metrics


__all__ = [
    "LocalizationModel",
    "LocalizationCriterion",
    "build_theta_grid",
]
