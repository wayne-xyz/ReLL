from __future__ import annotations

"""
Correlation visualization tools for ReLL translation search.

This script implements three position-finding methods from infer_visualize.py:

Method 1: Heatmap Cross-Correlation (Discrete Peak)
    - Finds the maximum value in the correlation heatmap
    - Returns integer pixel coordinates
    - Simple and fast, but limited to discrete grid positions

Method 2: Softmax Refinement (Expectation-Based)
    - Applies softmax to convert heatmap to probability distribution
    - Computes expectation (weighted average) for sub-pixel accuracy
    - Also provides uncertainty estimates (σx, σy)
    - Imported from: _softmax_refinement() in infer_visualize.py

Method 3: Gaussian Peak Fitting (Sub-pixel, 3×3 Parabolic)
    - Fits a parabolic surface to a 3×3 neighborhood around the peak
    - Uses simple, robust log-space quadratic fitting
    - Most accurate for sharp, well-defined peaks
    - Imported from: _gaussian_peak_fit_single() in infer_visualize.py
    - Note: fit_gaussian_surface() provides broader fit for visualization only

The script renders 2D contour panels comparing the three methods side-by-side.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure package imports work when executed directly
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.infer_visualize import (  # noqa: E402
    compute_embeddings_and_correlation,
    load_checkpoint,
    load_sample,
    _gaussian_peak_fit_single,
    _softmax_refinement,
)

def bilinear_sample(heatmap: torch.Tensor, x_offset: float, y_offset: float) -> float:
    """Sample heatmap at fractional (x, y) offsets relative to window center."""
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


@dataclass
class GaussianFitResult:
    surface: np.ndarray
    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    cov_xy: float
    amplitude: float
    baseline: float
    success: bool


def fit_gaussian_surface(
    heatmap: torch.Tensor,
    patch_radius: int = 5,
) -> GaussianFitResult:
    """
    Fit a full 2D Gaussian (with covariance) to the heatmap around its peak.

    Returns the reconstructed surface over the entire search window along with
    sub-pixel peak location and covariance statistics. Falls back to the
    original axis-aligned approximation if the least-squares fit is unstable.
    """
    heat_np = heatmap.cpu().numpy()
    H, W = heat_np.shape
    center_row = H // 2
    center_col = W // 2

    x_offsets = np.arange(W) - center_col
    y_offsets = np.arange(H) - center_row
    X, Y = np.meshgrid(x_offsets, y_offsets)

    peak_idx = int(np.argmax(heat_np))
    peak_row = peak_idx // W
    peak_col = peak_idx % W

    row_start = max(0, peak_row - patch_radius)
    row_end = min(H, peak_row + patch_radius + 1)
    col_start = max(0, peak_col - patch_radius)
    col_end = min(W, peak_col + patch_radius + 1)

    X_patch = X[row_start:row_end, col_start:col_end]
    Y_patch = Y[row_start:row_end, col_start:col_end]
    Z_patch = heat_np[row_start:row_end, col_start:col_end]

    if Z_patch.size < 6:
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    patch_min = float(Z_patch.min())
    eps = 1e-6
    if patch_min <= 0:
        offset = -patch_min + eps
    else:
        offset = eps
    Z_shifted = Z_patch + offset
    Z_shifted = np.clip(Z_shifted, eps, None)

    x = X_patch.reshape(-1).astype(np.float64)
    y = Y_patch.reshape(-1).astype(np.float64)
    z = Z_shifted.reshape(-1).astype(np.float64)
    log_z = np.log(z)

    weights = np.clip(z, eps, None)
    weights = weights**1.5
    weights /= max(weights.max(), eps)
    sqrt_w = np.sqrt(weights)

    design = np.stack(
        [
            np.ones_like(x),
            x,
            y,
            x**2,
            x * y,
            y**2,
        ],
        axis=1,
    )

    try:
        coeff, *_ = np.linalg.lstsq(design * sqrt_w[:, None], log_z * sqrt_w, rcond=None)
    except np.linalg.LinAlgError:
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    if not np.all(np.isfinite(coeff)):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    c0, c1, c2, c3, c4, c5 = coeff
    sigma_inv = np.array(
        [
            [-2.0 * c3, -c4],
            [-c4, -2.0 * c5],
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(sigma_inv)):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    det = float(np.linalg.det(sigma_inv))
    if det <= 1e-12 or sigma_inv[0, 0] <= 0.0 or sigma_inv[1, 1] <= 0.0:
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    try:
        mu_vec = np.linalg.solve(sigma_inv, np.array([c1, c2], dtype=np.float64))
    except np.linalg.LinAlgError:
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    if not np.all(np.isfinite(mu_vec)):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    peak_x_offset = x_offsets[peak_col]
    peak_y_offset = y_offsets[peak_row]
    if (abs(mu_vec[0] - peak_x_offset) > patch_radius + 1.5) or (
        abs(mu_vec[1] - peak_y_offset) > patch_radius + 1.5
    ):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    try:
        sigma = np.linalg.inv(sigma_inv)
    except np.linalg.LinAlgError:
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    if not np.all(np.isfinite(sigma)):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    sigma_x = float(np.sqrt(max(sigma[0, 0], 1e-9)))
    sigma_y = float(np.sqrt(max(sigma[1, 1], 1e-9)))
    cov_xy = float(sigma[0, 1])

    mu_col = mu_vec[0] + center_col
    mu_row = mu_vec[1] + center_row
    if mu_col < 0 or mu_col > (W - 1) or mu_row < 0 or mu_row > (H - 1):
        return _gaussian_surface_fallback(
            heatmap,
            heat_np,
            X,
            Y,
            patch_radius,
        )

    mu_vec_col = mu_vec.reshape(2, 1)
    log_amp = float(c0 + 0.5 * (mu_vec_col.T @ sigma_inv @ mu_vec_col))

    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    deltas = coords - mu_vec
    quad = np.einsum("ni,ij,nj->n", deltas, sigma_inv, deltas)
    surface_shifted = np.exp(log_amp - 0.5 * quad).reshape(H, W)
    surface = surface_shifted - offset
    surface = np.clip(surface, a_min=heat_np.min(), a_max=None)
    amplitude = max(float(surface_shifted.max() - offset), eps)
    baseline = float(surface.min())

    return GaussianFitResult(
        surface=surface,
        mu_x=float(mu_vec[0]),
        mu_y=float(mu_vec[1]),
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        cov_xy=cov_xy,
        amplitude=amplitude,
        baseline=baseline,
        success=True,
    )


def _gaussian_surface_fallback(
    heatmap: torch.Tensor,
    heat_np: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    patch_radius: int,
) -> GaussianFitResult:
    mu_x_px, mu_y_px = _gaussian_peak_fit_single(heatmap)
    gaussian_surface, sigma_x, sigma_y = _axis_aligned_gaussian_surface(
        heat_np,
        X,
        Y,
        mu_x_px,
        mu_y_px,
        patch_radius,
    )
    baseline = float(gaussian_surface.min())
    amplitude = float(gaussian_surface.max() - baseline)
    return GaussianFitResult(
        surface=gaussian_surface,
        mu_x=mu_x_px,
        mu_y=mu_y_px,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        cov_xy=0.0,
        amplitude=amplitude,
        baseline=baseline,
        success=False,
    )


def _gaussian_peak_fit_single_improved(heatmap: torch.Tensor) -> tuple[float, float]:
    """
    Enhanced sub-pixel refinement that fuses several estimators:

    1) Weighted centroid (robust baseline)
    2) 1D parabolic refinement along X/Y using bilinear samples
    3) 2D quadratic least-squares fit over the 3×3 neighborhood (raw values)
    4) Full Gaussian surface fit (radius 4) when stable

    The method selects the candidate with the highest interpolated correlation score,
    ensuring we never regress behind the robust weighted baseline while capturing sharper
    peaks when present.
    """
    import math

    H, W = heatmap.shape
    peak_idx = int(torch.argmax(heatmap.view(-1)))
    row = peak_idx // W
    col = peak_idx % W

    # Use exponential weighting for initial baseline (robust centroid)
    neighborhood = heatmap[max(0, row - 1) : min(H, row + 2), max(0, col - 1) : min(W, col + 2)]
    if neighborhood.numel() > 0:
        weights = torch.exp((neighborhood - neighborhood.max()) / 0.1)
        weights = weights / weights.sum()
        y_local = torch.arange(neighborhood.shape[0], dtype=heatmap.dtype, device=heatmap.device) - 1
        x_local = torch.arange(neighborhood.shape[1], dtype=heatmap.dtype, device=heatmap.device) - 1
        Y_grid, X_grid = torch.meshgrid(y_local, x_local, indexing='ij')
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
        # Quadratic interpolation along individual axes using bilinear samples.
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
        fy_center = fx_center  # already sampled at (base_x, base_y)
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
            coords = np.array(
                [
                    [-1.0, -1.0],
                    [0.0, -1.0],
                    [1.0, -1.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [-1.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            )

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

    # Larger context Gaussian surface fit (radius 4) for sharper peaks.
    try:
        surface_fit = fit_gaussian_surface(heatmap, patch_radius=4)
    except Exception:
        surface_fit = None
    if surface_fit is not None and surface_fit.success:
        add_candidate(surface_fit.mu_x, surface_fit.mu_y)

    # Blend high-scoring peak with softmax centroid when the heatmap is broad.
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
        return _gaussian_peak_fit_single(heatmap)

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

    if best_x is None or best_y is None:
        return _gaussian_peak_fit_single(heatmap)

    return float(best_x), float(best_y)


def _axis_aligned_gaussian_surface(
    heat_np: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mu_x_px: float,
    mu_y_px: float,
    patch_radius: int,
) -> tuple[np.ndarray, float, float]:
    shifted = heat_np - heat_np.min() + 1e-6
    X_rel = X - mu_x_px
    Y_rel = Y - mu_y_px
    mask = (np.abs(X_rel) <= patch_radius) & (np.abs(Y_rel) <= patch_radius)

    X_sel = X_rel[mask].reshape(-1)
    Y_sel = Y_rel[mask].reshape(-1)
    Z_sel = shifted[mask].reshape(-1)
    valid = Z_sel > 0

    if valid.sum() < 5:
        default_sigma = max(2.0, patch_radius / 2.0)
        gaussian_surface = np.exp(
            -((X_rel) ** 2) / (2 * default_sigma**2) - ((Y_rel) ** 2) / (2 * default_sigma**2)
        )
        return gaussian_surface, default_sigma, default_sigma

    X_sel = X_sel[valid]
    Y_sel = Y_sel[valid]
    Z_sel = Z_sel[valid]

    A = np.stack(
        [
            np.ones_like(X_sel),
            -(X_sel**2),
            -(Y_sel**2),
        ],
        axis=1,
    )
    b = np.log(Z_sel)

    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    log_amp, cx, cy = coeff

    sigma_x_sq = 1.0 / max(1e-9, -2.0 * cx) if cx < 0 else (patch_radius**2)
    sigma_y_sq = 1.0 / max(1e-9, -2.0 * cy) if cy < 0 else (patch_radius**2)
    sigma_x = float(np.sqrt(sigma_x_sq))
    sigma_y = float(np.sqrt(sigma_y_sq))

    gaussian_surface = np.exp(
        log_amp
        - (X_rel**2) / (2.0 * sigma_x_sq)
        - (Y_rel**2) / (2.0 * sigma_y_sq)
    )

    return gaussian_surface, sigma_x, sigma_y


def plot_correlation_2d(
    heatmap: torch.Tensor,
    softmax_probs: torch.Tensor,
    softmax_mu: tuple[float, float],
    softmax_sigma: tuple[float, float],
    gaussian_mu: tuple[float, float],
    gaussian_surface: np.ndarray,
    gaussian_sigma: tuple[float, float],
    resolution: float,
    use_meters: bool,
    peak_offset: tuple[int, int],
    peak_score: float,
    softmax_score: float,
    gaussian_score: float,
    gaussian_success: bool,
    gaussian_cov_xy: float = 0.0,
    save_path: Path | None = None,
) -> None:
    """
    Render 2D contour comparison for the three position-finding methods:
    1. Heatmap cross-correlation (discrete peak)
    2. Softmax refinement (expectation-based)
    3. Gaussian peak fitting (sub-pixel fitting)
    """
    heat_np = heatmap.cpu().numpy()
    softmax_np = softmax_probs.cpu().numpy()

    H, W = heat_np.shape
    center_row = H // 2
    center_col = W // 2
    x_offsets_px = np.arange(W) - center_col
    y_offsets_px = np.arange(H) - center_row
    X_px, Y_px = np.meshgrid(x_offsets_px, y_offsets_px)

    if use_meters:
        X_plot = X_px * resolution
        Y_plot = Y_px * resolution
        unit = "m"
    else:
        X_plot = X_px
        Y_plot = Y_px
        unit = "px"

    softmax_sigma_x, softmax_sigma_y = softmax_sigma
    gaussian_sigma_x, gaussian_sigma_y = gaussian_sigma

    if use_meters:
        softmax_sigma_x_disp = softmax_sigma_x * resolution
        softmax_sigma_y_disp = softmax_sigma_y * resolution
        gaussian_sigma_x_disp = gaussian_sigma_x * resolution
        gaussian_sigma_y_disp = gaussian_sigma_y * resolution
    else:
        softmax_sigma_x_disp = softmax_sigma_x
        softmax_sigma_y_disp = softmax_sigma_y
        gaussian_sigma_x_disp = gaussian_sigma_x
        gaussian_sigma_y_disp = gaussian_sigma_y

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # === Method 1: Raw Heatmap Cross-Correlation (Discrete Peak) ===
    im0 = axes[0].imshow(
        heat_np,
        cmap="coolwarm",
        origin="lower",
        extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
    )
    axes[0].set_title(
        f"Method 1: Heatmap Cross-Correlation\nPeak Score = {peak_score:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel(f"ΔX ({unit})")
    axes[0].set_ylabel(f"ΔY ({unit})")
    # Plot discrete peak
    axes[0].scatter(
        [peak_offset[0] * resolution if use_meters else peak_offset[0]],
        [peak_offset[1] * resolution if use_meters else peak_offset[1]],
        c="gold",
        s=140,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label=f"Discrete Peak ({peak_offset[0]:+.1f}{unit}, {peak_offset[1]:+.1f}{unit})",
    )
    # Also show refined positions for comparison
    sx, sy = softmax_mu
    gx, gy = gaussian_mu
    axes[0].scatter(
        [sx * resolution if use_meters else sx],
        [sy * resolution if use_meters else sy],
        c="red",
        marker="^",
        s=90,
        edgecolors="white",
        linewidths=1.0,
        alpha=0.7,
        label=f"Softmax μ ({sx:+.2f}{unit}, {sy:+.2f}{unit})",
    )
    axes[0].scatter(
        [gx * resolution if use_meters else gx],
        [gy * resolution if use_meters else gy],
        facecolors="none",
        edgecolors="yellow",
        marker="s",
        s=100,
        linewidths=1.5,
        alpha=0.7,
        label=f"Gaussian μ ({gx:+.2f}{unit}, {gy:+.2f}{unit})",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Correlation")

    # === Method 2: Softmax Refinement (Expectation-Based) ===
    softmax_display = np.log(softmax_np + 1e-8)
    cf1 = axes[1].contourf(
        X_plot,
        Y_plot,
        softmax_display,
        levels=40,
        cmap="magma",
    )
    axes[1].scatter(
        [sx * resolution if use_meters else sx],
        [sy * resolution if use_meters else sy],
        c="red",
        marker="^",
        s=120,
        edgecolors="white",
        linewidths=1.5,
        label=f"Softmax μ ({sx:+.2f}{unit}, {sy:+.2f}{unit})",
    )
    # Show discrete peak for reference
    axes[1].scatter(
        [peak_offset[0] * resolution if use_meters else peak_offset[0]],
        [peak_offset[1] * resolution if use_meters else peak_offset[1]],
        c="gold",
        s=100,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.5,
        label=f"Discrete Peak",
    )
    axes[1].set_title(
        f"Method 2: Softmax Refinement\nScore ≈ {softmax_score:.4f}, σ≈({softmax_sigma_x_disp:.2f},{softmax_sigma_y_disp:.2f}) {unit}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel(f"ΔX ({unit})")
    axes[1].set_ylabel(f"ΔY ({unit})")
    axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04, label="log Probability")

    # === Method 3: Gaussian Peak Fitting (Sub-pixel) with Surface Projection ===
    # Normalize Gaussian surface for better visualization (not log, actual surface)
    gaussian_normalized = (gaussian_surface - gaussian_surface.min()) / (gaussian_surface.max() - gaussian_surface.min() + 1e-9)

    # Create filled contour plot showing the Gaussian surface
    cf2 = axes[2].contourf(
        X_plot,
        Y_plot,
        gaussian_normalized,
        levels=30,
        cmap="viridis",
        alpha=0.8,
    )

    # Add contour lines for additional depth cues
    contour_lines = axes[2].contour(
        X_plot,
        Y_plot,
        gaussian_normalized,
        levels=10,
        colors="white",
        linewidths=0.5,
        alpha=0.4,
    )

    # Draw confidence ellipses (1σ, 2σ, 3σ) based on the fitted Gaussian covariance
    from matplotlib.patches import Ellipse

    gx_plot = gx * resolution if use_meters else gx
    gy_plot = gy * resolution if use_meters else gy

    # Create covariance matrix from fitted parameters
    cov_matrix = np.array([
        [gaussian_sigma_x**2, gaussian_cov_xy],
        [gaussian_cov_xy, gaussian_sigma_y**2]
    ])

    # Compute eigenvalues and eigenvectors for ellipse orientation
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Compute rotation angle from eigenvectors
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad)

    # Ellipse width/height are 2*sqrt(eigenvalue) for each axis
    ellipse_width_base = 2 * np.sqrt(max(eigenvalues[0], 1e-9))
    ellipse_height_base = 2 * np.sqrt(max(eigenvalues[1], 1e-9))

    # Draw confidence ellipses at 1σ, 2σ, 3σ
    confidence_levels = [1, 2, 3]
    colors_ellipse = ['yellow', 'orange', 'red']
    alphas_ellipse = [0.7, 0.5, 0.3]

    for level, color, alpha in zip(confidence_levels, colors_ellipse, alphas_ellipse):
        # Scale by confidence level
        width = level * ellipse_width_base
        height = level * ellipse_height_base

        if use_meters:
            width *= resolution
            height *= resolution

        ellipse = Ellipse(
            xy=(gx_plot, gy_plot),
            width=width,
            height=height,
            angle=angle_deg,  # Rotation angle from covariance
            facecolor="none",
            edgecolor=color,
            linewidth=2 if level == 1 else 1.5,
            linestyle="--" if level > 1 else "-",
            alpha=alpha,
            label=f"{level}σ ellipse" if level == 1 else None,
        )
        axes[2].add_patch(ellipse)

    # Mark the Gaussian peak
    axes[2].scatter(
        [gx_plot],
        [gy_plot],
        facecolors="yellow",
        edgecolors="white",
        marker="s",
        s=150,
        linewidths=2.5,
        label=f"Gaussian μ ({gx:+.2f}{unit}, {gy:+.2f}{unit})",
        zorder=10,
    )

    # Show discrete peak for reference
    axes[2].scatter(
        [peak_offset[0] * resolution if use_meters else peak_offset[0]],
        [peak_offset[1] * resolution if use_meters else peak_offset[1]],
        c="gold",
        s=100,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.5,
        label=f"Discrete Peak",
        zorder=9,
    )

    axes[2].set_title(
        f"Method 3: Gaussian Peak Fitting (3×3 Parabolic)\nScore ≈ {gaussian_score:.4f} (Surface fit σ≈({gaussian_sigma_x_disp:.2f},{gaussian_sigma_y_disp:.2f}) {unit})",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel(f"ΔX ({unit})")
    axes[2].set_ylabel(f"ΔY ({unit})")
    axes[2].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_aspect('equal', adjustable='box')
    fig.colorbar(cf2, ax=axes[2], fraction=0.046, pad=0.04, label="Normalized Gaussian Surface")

    # Overall title
    fig.suptitle(
        "Three Position-Finding Methods Comparison (2D Mode)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Save] 2D contour figure saved to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ReLL translation cross-correlation with detailed 2D contour panels."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"G:\GithubProject\ReLL\model-save\best_1000.ckpt"),
        help="Path to model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(r"Rell-sample-raster2\0As5GhcTExRFWNboqeHwpegdeGf2j0YW_036"),
        help="Path to sample directory (containing .npy files).",
    )
    parser.add_argument(
        "--lidar-variant",
        type=str,
        default="gicp",
        choices=["gicp", "non_aligned"],
        help="LiDAR variant to use for correlation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the visualization (e.g., output.png).",
    )
    parser.add_argument(
        "--meters",
        action="store_true",
        help="Plot X/Y axes in meters instead of pixels.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show visualization GUI. Without this flag, only prints results.",
    )
    parser.add_argument(
        "--gaussian-method",
        type=str,
        default="improved",
        choices=["improved"],
        help="Gaussian fitting method: improved (multi-stage refinement with Gaussian surface fit).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReLL Cross-Correlation Visualizer")
    print("=" * 70)

    # Load trained model
    model, model_cfg = load_checkpoint(args.checkpoint, device=args.device)

    # Load single sample (no augmentation)
    sample = load_sample(args.sample, lidar_variant=args.lidar_variant)
    lidar = sample["lidar"].to(args.device)
    geospatial = sample["map"].to(args.device)
    resolution = sample["resolution"]

    print("\n[Inference] Computing correlation heatmap...")
    results = compute_embeddings_and_correlation(model, lidar, geospatial)
    heatmap = results["correlation_heatmap"]
    radius_px = results.get("correlation_radius_px", heatmap.shape[-1] // 2)

    # Extract discrete peak info (already centered grid)
    peak_idx = int(torch.argmax(heatmap))
    H, W = heatmap.shape
    peak_row = peak_idx // W
    peak_col = peak_idx % W
    center_row = H // 2
    center_col = W // 2
    peak_y_offset = peak_row - center_row
    peak_x_offset = peak_col - center_col
    peak_score = float(heatmap[peak_row, peak_col].item())

    (
        softmax_probs,
        softmax_mu_x_px,
        softmax_mu_y_px,
        softmax_sigma_x_px,
        softmax_sigma_y_px,
    ) = _softmax_refinement(heatmap)

    # Select Gaussian fitting method based on args
    gaussian_mu_x_px, gaussian_mu_y_px = _gaussian_peak_fit_single_improved(heatmap)

    # For visualization, also compute the full surface fit with larger patch
    # This is only for showing the Gaussian surface, not for position accuracy
    fit_patch_radius = max(3, min(radius_px, 6))
    gaussian_fit = fit_gaussian_surface(heatmap, patch_radius=fit_patch_radius)
    gaussian_sigma_x_px = gaussian_fit.sigma_x
    gaussian_sigma_y_px = gaussian_fit.sigma_y

    softmax_score = bilinear_sample(heatmap, softmax_mu_x_px, softmax_mu_y_px)
    gaussian_score = bilinear_sample(heatmap, gaussian_mu_x_px, gaussian_mu_y_px)

    # Calculate distances from ground truth (0, 0) for final summary
    import math
    peak_dist_px = math.sqrt(peak_x_offset**2 + peak_y_offset**2)
    peak_dist_m = peak_dist_px * resolution
    softmax_dist_px = math.sqrt(softmax_mu_x_px**2 + softmax_mu_y_px**2)
    softmax_dist_m = softmax_dist_px * resolution
    gaussian_dist_px = math.sqrt(gaussian_mu_x_px**2 + gaussian_mu_y_px**2)
    gaussian_dist_m = gaussian_dist_px * resolution

    print("\n" + "=" * 70)
    print("Three Position-Finding Methods Summary")
    print("=" * 70)
    print(
        f"[Method 1: Heatmap Cross-Correlation]\n"
        f"  Discrete Peak: ({peak_x_offset:+.1f}px, {peak_y_offset:+.1f}px) = "
        f"({peak_x_offset * resolution:+.3f}m, {peak_y_offset * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {peak_dist_px:.3f}px = {peak_dist_m:.4f}m\n"
        f"  Score: {peak_score:.4f}"
    )
    print(
        f"\n[Method 2: Softmax Refinement]\n"
        f"  Refined μ: ({softmax_mu_x_px:+.3f}px, {softmax_mu_y_px:+.3f}px) = "
        f"({softmax_mu_x_px * resolution:+.3f}m, {softmax_mu_y_px * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {softmax_dist_px:.3f}px = {softmax_dist_m:.4f}m\n"
        f"  σ: ({softmax_sigma_x_px:.3f}px, {softmax_sigma_y_px:.3f}px) = "
        f"({softmax_sigma_x_px * resolution:.3f}m, {softmax_sigma_y_px * resolution:.3f}m)\n"
        f"  Score: {softmax_score:.4f}"
    )

    # Determine if Gaussian beats baselines
    gaussian_beats_peak = "✓" if gaussian_dist_px < peak_dist_px else "✗"
    gaussian_beats_softmax = "✓" if gaussian_dist_px < softmax_dist_px else "✗"
    gaussian_best = " 🏆 BEST!" if gaussian_dist_px < min(peak_dist_px, softmax_dist_px) else ""

    print(
        f"\n[Method 3: Gaussian Peak Fitting ({args.gaussian_method})]\n"
        f"  Refined μ: ({gaussian_mu_x_px:+.3f}px, {gaussian_mu_y_px:+.3f}px) = "
        f"({gaussian_mu_x_px * resolution:+.3f}m, {gaussian_mu_y_px * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {gaussian_dist_px:.3f}px = {gaussian_dist_m:.4f}m "
        f"[vs Peak:{gaussian_beats_peak} vs Softmax:{gaussian_beats_softmax}]{gaussian_best}\n"
        f"  Score: {gaussian_score:.4f}\n"
        f"  (Surface σ for viz: ({gaussian_sigma_x_px:.3f}px, {gaussian_sigma_y_px:.3f}px), cov={gaussian_fit.cov_xy:+.4f})"
    )
    print("=" * 70)

    if args.vis:
        plot_correlation_2d(
            heatmap,
            softmax_probs,
            (softmax_mu_x_px, softmax_mu_y_px),
            (softmax_sigma_x_px, softmax_sigma_y_px),
            (gaussian_mu_x_px, gaussian_mu_y_px),
            gaussian_fit.surface,
            (gaussian_sigma_x_px, gaussian_sigma_y_px),
            resolution,
            use_meters=args.meters,
            peak_offset=(peak_x_offset, peak_y_offset),
            peak_score=peak_score,
            softmax_score=softmax_score,
            gaussian_score=gaussian_score,
            gaussian_success=gaussian_fit.success,
            gaussian_cov_xy=gaussian_fit.cov_xy,
            save_path=args.save,
        )
    else:
        print("\n[Info] Skipping visualization (use --vis to show plots)")
        print("[Done] Results printed above.")


if __name__ == "__main__":
    main()
