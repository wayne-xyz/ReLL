from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def bilinear_sample(heatmap: torch.Tensor, x_offset: float, y_offset: float) -> float:
    """
    Sample a 2D heatmap at fractional offsets relative to its center index.

    Offsets are given in pixels with (0, 0) at the window center. Values are
    bilinearly interpolated to provide a smooth surface around discrete grid points.
    """
    H, W = heatmap.shape
    center_row = H // 2
    center_col = W // 2
    x = x_offset + center_col
    y = y_offset + center_row

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = int(np.clip(x0, 0, W - 1))
    x1 = int(np.clip(x1, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    y1 = int(np.clip(y1, 0, H - 1))

    q11 = heatmap[y0, x0].item()
    q21 = heatmap[y0, x1].item()
    q12 = heatmap[y1, x0].item()
    q22 = heatmap[y1, x1].item()

    x_frac = x - x0
    y_frac = y - y0

    top = q11 * (1 - x_frac) + q21 * x_frac
    bottom = q12 * (1 - x_frac) + q22 * x_frac
    value = top * (1 - y_frac) + bottom * y_frac
    return float(value)


@dataclass(frozen=True)
class GaussianPeakResult:
    """Outputs of the refined Gaussian-style peak finder."""

    x: float
    y: float
    score: float


def _iter_quadratic_coords() -> Iterable[tuple[float, float]]:
    """3×3 neighborhood coordinates around the peak used for quadratic fits."""
    for dy in (-1.0, 0.0, 1.0):
        for dx in (-1.0, 0.0, 1.0):
            yield dx, dy


def _softmax_refinement(heatmap: torch.Tensor) -> Tuple[torch.Tensor, float, float, float, float]:
    """
    Compute softmax probabilities, expectation, and standard deviations in pixel space.
    Matches the refinement used during inference/visualization pipelines.
    """
    probs = F.softmax(heatmap.view(-1), dim=0).view_as(heatmap)
    H, W = probs.shape
    center_row = H // 2
    center_col = W // 2

    grid_y = torch.arange(H, dtype=probs.dtype, device=probs.device) - center_row
    grid_x = torch.arange(W, dtype=probs.dtype, device=probs.device) - center_col
    coord_y = grid_y.view(-1, 1).expand(H, W)
    coord_x = grid_x.view(1, -1).expand(H, W)

    mu_x = (probs * coord_x).sum()
    mu_y = (probs * coord_y).sum()

    dx = coord_x - mu_x
    dy = coord_y - mu_y
    var_x = (probs * dx.pow(2)).sum()
    var_y = (probs * dy.pow(2)).sum()

    sigma_x = float(torch.sqrt(var_x.clamp(min=1e-9)).item())
    sigma_y = float(torch.sqrt(var_y.clamp(min=1e-9)).item())

    return probs.cpu(), float(mu_x.item()), float(mu_y.item()), sigma_x, sigma_y


def _gaussian_peak_fit_single(heatmap: torch.Tensor) -> Tuple[float, float]:
    """
    Apply the compact Gaussian peak fit over a 3×3 neighborhood around the discrete peak.
    Provides a stable fallback when the improved refinement fails.
    """
    H, W = heatmap.shape
    peak_idx = int(torch.argmax(heatmap.view(-1)))
    row = peak_idx // W
    col = peak_idx % W

    row_start = max(0, row - 1)
    row_end = min(H, row + 2)
    col_start = max(0, col - 1)
    col_end = min(W, col + 2)
    neighborhood = heatmap[row_start:row_end, col_start:col_end]

    if neighborhood.shape != (3, 3):
        return float(col - W // 2), float(row - H // 2)

    log_neighborhood = torch.log(neighborhood.clamp(min=1e-10))
    c_00 = float(log_neighborhood[1, 1].item())
    c_m10 = float(log_neighborhood[1, 0].item())
    c_p10 = float(log_neighborhood[1, 2].item())
    c_0m1 = float(log_neighborhood[0, 1].item())
    c_0p1 = float(log_neighborhood[2, 1].item())

    d2_dx2 = c_m10 - 2.0 * c_00 + c_p10
    d2_dy2 = c_0m1 - 2.0 * c_00 + c_0p1
    d_dx = (c_p10 - c_m10) * 0.5
    d_dy = (c_0p1 - c_0m1) * 0.5

    dx = 0.0
    dy = 0.0
    if d2_dx2 < -1e-6:
        dx = max(-1.0, min(1.0, -d_dx / d2_dx2))
    if d2_dy2 < -1e-6:
        dy = max(-1.0, min(1.0, -d_dy / d2_dy2))

    refined_x = (col - W // 2) + dx
    refined_y = (row - H // 2) + dy

    return float(refined_x), float(refined_y)


def _fallback_result(heatmap: torch.Tensor) -> GaussianPeakResult:
    x_fallback, y_fallback = _gaussian_peak_fit_single(heatmap)
    score = bilinear_sample(heatmap, x_fallback, y_fallback)
    return GaussianPeakResult(float(x_fallback), float(y_fallback), float(score))


def gaussian_peak_refine(heatmap: torch.Tensor) -> GaussianPeakResult:
    """
    Refine the correlation peak using a blend of centroid, quadratic, and Newton steps.
    """
    H, W = heatmap.shape
    peak_idx = int(torch.argmax(heatmap.view(-1)))
    row = peak_idx // W
    col = peak_idx % W

    # Weighted centroid baseline around the discrete peak.
    neighborhood = heatmap[max(0, row - 1) : min(H, row + 2), max(0, col - 1) : min(W, col + 2)]
    if neighborhood.numel() > 0:
        weights = torch.exp((neighborhood - neighborhood.max()) / 0.1)
        weights = weights / weights.sum()
        y_local = torch.arange(neighborhood.shape[0], dtype=heatmap.dtype, device=heatmap.device) - 1
        x_local = torch.arange(neighborhood.shape[1], dtype=heatmap.dtype, device=heatmap.device) - 1
        Y_grid, X_grid = torch.meshgrid(y_local, x_local, indexing="ij")
        dx = (weights * X_grid).sum().item()
        dy = (weights * Y_grid).sum().item()
        weighted_x = (col - W // 2) + dx
        weighted_y = (row - H // 2) + dy
    else:
        weighted_x = float(col - W // 2)
        weighted_y = float(row - H // 2)

    candidates: list[tuple[float, float, float]] = []

    def add_candidate(x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        score = bilinear_sample(heatmap, x, y)
        candidates.append((score, x, y))

    add_candidate(weighted_x, weighted_y)

    def one_dim_refine(base_x: float, base_y: float) -> tuple[float, float]:
        fx_minus = bilinear_sample(heatmap, base_x - 1.0, base_y)
        fx_center = bilinear_sample(heatmap, base_x, base_y)
        fx_plus = bilinear_sample(heatmap, base_x + 1.0, base_y)
        denom_x = fx_minus - 2.0 * fx_center + fx_plus
        if abs(denom_x) > 1e-9:
            delta_x = 0.5 * (fx_minus - fx_plus) / denom_x
            delta_x = float(np.clip(delta_x, -0.75, 0.75))
        else:
            delta_x = 0.0

        fy_minus = bilinear_sample(heatmap, base_x, base_y - 1.0)
        fy_center = fx_center
        fy_plus = bilinear_sample(heatmap, base_x, base_y + 1.0)
        denom_y = fy_minus - 2.0 * fy_center + fy_plus
        if abs(denom_y) > 1e-9:
            delta_y = 0.5 * (fy_minus - fy_plus) / denom_y
            delta_y = float(np.clip(delta_y, -0.75, 0.75))
        else:
            delta_y = 0.0

        return base_x + delta_x, base_y + delta_y

    refined_weighted_x, refined_weighted_y = one_dim_refine(weighted_x, weighted_y)
    add_candidate(refined_weighted_x, refined_weighted_y)

    def newton_refine(x_start: float, y_start: float) -> tuple[float, float]:
        x = float(x_start)
        y = float(y_start)
        step = 0.5
        for _ in range(3):
            f0 = bilinear_sample(heatmap, x, y)
            fx_plus = bilinear_sample(heatmap, x + step, y)
            fx_minus = bilinear_sample(heatmap, x - step, y)
            fy_plus = bilinear_sample(heatmap, x, y + step)
            fy_minus = bilinear_sample(heatmap, x, y - step)

            grad_x = (fx_plus - fx_minus) / (2.0 * step)
            grad_y = (fy_plus - fy_minus) / (2.0 * step)

            fxx = (fx_plus - 2.0 * f0 + fx_minus) / (step**2)
            fyy = (fy_plus - 2.0 * f0 + fy_minus) / (step**2)
            fxy = (
                bilinear_sample(heatmap, x + step, y + step)
                - bilinear_sample(heatmap, x + step, y - step)
                - bilinear_sample(heatmap, x - step, y + step)
                + bilinear_sample(heatmap, x - step, y - step)
            ) / (4.0 * step**2)

            hessian = np.array([[fxx, fxy], [fxy, fyy]], dtype=np.float64)
            gradient = np.array([grad_x, grad_y], dtype=np.float64)

            if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
                break

            eigvals = np.linalg.eigvalsh(hessian)
            if eigvals.max() >= -1e-6:
                hessian = hessian - np.eye(2, dtype=np.float64) * (eigvals.max() + 1e-6)

            det = float(np.linalg.det(hessian))
            if abs(det) < 1e-9:
                break

            try:
                delta = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                break

            if not np.all(np.isfinite(delta)):
                break

            delta = np.clip(delta, -1.0, 1.0)
            x += float(delta[0])
            y += float(delta[1])

            if abs(delta[0]) < 1e-3 and abs(delta[1]) < 1e-3:
                break

            x = float(np.clip(x, -(W // 2), W // 2))
            y = float(np.clip(y, -(H // 2), H // 2))

        return x, y

    newton_x, newton_y = newton_refine(weighted_x, weighted_y)
    add_candidate(newton_x, newton_y)

    softmax_mu_x = softmax_mu_y = None
    softmax_sigma_x = softmax_sigma_y = None
    try:
        _, softmax_mu_x, softmax_mu_y, softmax_sigma_x, softmax_sigma_y = _softmax_refinement(heatmap)
        add_candidate(float(softmax_mu_x), float(softmax_mu_y))
    except Exception:
        softmax_mu_x = softmax_mu_y = None

    if 0 < row < H - 1 and 0 < col < W - 1:
        neighborhood = heatmap[row - 1 : row + 2, col - 1 : col + 2]
        patch = neighborhood.detach().cpu().to(dtype=torch.float64).numpy()
        if np.all(np.isfinite(patch)):
            coords = np.array(list(_iter_quadratic_coords()), dtype=np.float64)
            design = np.column_stack(
                [
                    np.ones(len(coords), dtype=np.float64),
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 0] ** 2,
                    coords[:, 0] * coords[:, 1],
                    coords[:, 1] ** 2,
                ]
            )

            try:
                coeff, *_ = np.linalg.lstsq(design, patch.reshape(-1), rcond=None)
            except np.linalg.LinAlgError:
                coeff = None

            if coeff is not None and np.all(np.isfinite(coeff)):
                _, c1, c2, c3, c4, c5 = coeff
                hessian = np.array([[2.0 * c3, c4], [c4, 2.0 * c5]], dtype=np.float64)
                gradient = np.array([c1, c2], dtype=np.float64)
                eigvals = np.linalg.eigvalsh(hessian)
                if eigvals.max() < -1e-9:
                    try:
                        offset = np.linalg.solve(hessian, -gradient)
                    except np.linalg.LinAlgError:
                        offset = None
                    if offset is not None and np.all(np.isfinite(offset)):
                        dx, dy = float(np.clip(offset[0], -1.25, 1.25)), float(np.clip(offset[1], -1.25, 1.25))
                        base_x = col - W // 2
                        base_y = row - H // 2
                        add_candidate(base_x + dx, base_y + dy)

    if softmax_mu_x is not None and candidates:
        best_raw_score, best_raw_x, best_raw_y = max(candidates, key=lambda item: item[0])
        sigma_avg = float(max(0.25, (softmax_sigma_x + softmax_sigma_y) * 0.5))
        alpha_primary = float(np.clip(0.45 + 0.35 * (sigma_avg / 0.8), 0.0, 0.92))
        blended_primary_x = (1.0 - alpha_primary) * best_raw_x + alpha_primary * float(softmax_mu_x)
        blended_primary_y = (1.0 - alpha_primary) * best_raw_y + alpha_primary * float(softmax_mu_y)
        add_candidate(blended_primary_x, blended_primary_y)

        alpha_secondary = alpha_primary * 0.5
        blended_secondary_x = (1.0 - alpha_secondary) * best_raw_x + alpha_secondary * float(softmax_mu_x)
        blended_secondary_y = (1.0 - alpha_secondary) * best_raw_y + alpha_secondary * float(softmax_mu_y)
        add_candidate(blended_secondary_x, blended_secondary_y)

    if not candidates:
        return _fallback_result(heatmap)

    penalty_coeff = 0.22
    if softmax_sigma_x is not None and softmax_sigma_y is not None:
        sigma_avg = float(max(0.2, (softmax_sigma_x + softmax_sigma_y) * 0.5))
        penalty_coeff = float(np.clip(0.18 + 0.52 * (sigma_avg / 0.9), 0.18, 0.88))

    best_score, best_x, best_y = None, None, None
    best_objective = None
    for score, x, y in candidates:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        penalty = penalty_coeff * (x * x + y * y)
        objective = score - penalty
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_score, best_x, best_y = score, x, y

    if best_x is None or best_y is None or best_score is None:
        return _fallback_result(heatmap)

    return GaussianPeakResult(float(best_x), float(best_y), float(best_score))


__all__ = ["GaussianPeakResult", "bilinear_sample", "gaussian_peak_refine"]
