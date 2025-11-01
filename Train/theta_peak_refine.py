from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ThetaPeakResult:
    """Outputs of the theta peak finder."""

    theta_deg: float
    sigma_deg: float
    confidence: float
    prob_distribution: torch.Tensor  # Softmax probabilities for visualization


def theta_peak_refine(
    theta_logits: torch.Tensor,
    theta_candidates_deg: torch.Tensor,
) -> ThetaPeakResult:
    """
    Refine the rotation peak using softmax expectation.

    This function finds the best rotation angle from the correlation scores
    across different rotation candidates. It uses softmax to convert logits
    into a probability distribution and computes the expected value (mean)
    and uncertainty (standard deviation).

    Args:
        theta_logits: (num_angles,) correlation scores for each rotation candidate
        theta_candidates_deg: (num_angles,) rotation angles in degrees

    Returns:
        ThetaPeakResult containing:
            - theta_deg: predicted rotation angle in degrees
            - sigma_deg: uncertainty (standard deviation) in degrees
            - confidence: peak confidence score (0-1, based on max probability)
            - prob_distribution: softmax probability distribution for visualization
    """
    # Ensure 1D tensors
    if theta_logits.dim() != 1:
        raise ValueError(f"theta_logits must be 1D, got shape {theta_logits.shape}")
    if theta_candidates_deg.dim() != 1:
        raise ValueError(f"theta_candidates_deg must be 1D, got shape {theta_candidates_deg.shape}")
    if theta_logits.shape[0] != theta_candidates_deg.shape[0]:
        raise ValueError(
            f"Shape mismatch: theta_logits {theta_logits.shape} vs "
            f"theta_candidates_deg {theta_candidates_deg.shape}"
        )

    # Apply softmax to get probability distribution
    theta_prob = F.softmax(theta_logits, dim=0)

    # Compute expected rotation angle (weighted average)
    theta_pred_deg = (theta_prob * theta_candidates_deg).sum()

    # Compute variance and standard deviation
    theta_diff = theta_candidates_deg - theta_pred_deg
    var_theta = (theta_prob * (theta_diff ** 2)).sum()
    sigma_theta_deg = torch.sqrt(var_theta.clamp(min=1e-9))

    # Compute confidence as the maximum probability
    # High max prob = sharp peak = confident
    # Low max prob = broad distribution = uncertain
    confidence = theta_prob.max()

    return ThetaPeakResult(
        theta_deg=float(theta_pred_deg.item()),
        sigma_deg=float(sigma_theta_deg.item()),
        confidence=float(confidence.item()),
        prob_distribution=theta_prob.cpu(),
    )


def theta_peak_refine_batch(
    theta_logits: torch.Tensor,
    theta_candidates_deg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version for training/evaluation. Returns tensors instead of dataclass.

    Args:
        theta_logits: (B, num_angles) correlation scores
        theta_candidates_deg: (num_angles,) rotation angles in degrees

    Returns:
        theta_pred_deg: (B,) predicted rotation angles
        sigma_theta_deg: (B,) uncertainties
    """
    if theta_logits.dim() != 2:
        raise ValueError(f"theta_logits must be 2D for batch, got shape {theta_logits.shape}")

    # Apply softmax across angles dimension
    theta_prob = F.softmax(theta_logits, dim=-1)  # (B, num_angles)

    # Compute expected rotation angle
    theta_pred_deg = (theta_prob * theta_candidates_deg.view(1, -1)).sum(dim=-1)  # (B,)

    # Compute variance and standard deviation
    theta_diff = theta_candidates_deg.view(1, -1) - theta_pred_deg.view(-1, 1)  # (B, num_angles)
    var_theta = (theta_prob * (theta_diff ** 2)).sum(dim=-1)  # (B,)
    sigma_theta_deg = torch.sqrt(var_theta.clamp(min=1e-9))  # (B,)

    return theta_pred_deg, sigma_theta_deg


__all__ = ["ThetaPeakResult", "theta_peak_refine", "theta_peak_refine_batch"]
