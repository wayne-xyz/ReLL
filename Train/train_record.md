## 3.3 Record Experence  :
Dataset sample 1100, batch size 64

Before op:
| Metric | Train | Validation |
|:--|:--|:--|
| **Loss** | 1.9930 | 1.9774 |
| **Mean Abs XY (m)** | 1.963 | 1.953 |
| **Mean Abs Theta (rad)** | 0.0297 | 0.0244 |

### Timing (Per-Batch Averages)

| Phase | Fetch (ms) | To Device (ms) | Forward (ms) | Loss (ms) | Backward (ms) | Step (ms) | Batches |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Train** | 49.3 | 18.0 | 1950.2 | 4.3 | 5521.7 | 1.4 | 15 |
| **Eval** | 171.5 | 16.8 | 1830.8 | 3.5 | — | — | 4 |

### Epoch Timing Summary

| Epoch Wall Time (s) | Eval Total (s) | Checkpoint (s) | Remaining (approx.) | ETA |
|--:|--:|--:|--:|:--|
| 121.27 | 8.09 | 0.04 | 12m 07s | 2025-10-26 02:03:53 |


after op


Epoch 06/10: train_loss=1.6216  | train_mean_abs_xy=1.590 m  | train_mean_abs_theta=0.0318 rad  ||  val_loss=1.6273  | val_mean_abs_xy=1.586 m  | val_mean_abs_theta=0.0413 rad  
  ⏱️ Timing (per-batch averages): fetch=3.9 ms, to_dev=4.8 ms, forward=91.6 ms, loss=4.4 ms, backward=1225.6 ms, step=1.5 ms  (batches=59)
  ⏱️ Eval (per-batch averages): fetch=13.3 ms, to_dev=4.7 ms, forward=91.1 ms, loss=3.8 ms  (batches=15)
  ⏲️ Epoch wall time=80.28s  | eval total=1.70s  | ckpt=0.04s  | remaining≈5m 21s  (ETA ~ 2025-10-26 04:40:25)

opt2
Epoch 15/20: train_loss=3.7238  | train_mean_abs_xy=1.842 m  | train_mean_abs_theta=0.0406 rad  ||  val_loss=3.7136  | val_mean_abs_xy=1.835 m  | val_mean_abs_theta=0.0439 rad  
  ⏱️ Timing (per-batch averages): fetch=2.8 ms, to_dev=4.7 ms, forward=88.2 ms, loss=4.4 ms, backward=1201.2 ms, step=1.9 ms  (batches=59)
  ⏱️ Eval (per-batch averages): fetch=14.3 ms, to_dev=4.7 ms, forward=87.5 ms, loss=3.8 ms  (batches=15)
  ⏲️ Epoch wall time=78.55s  | eval total=1.66s  | ckpt=0.03s  | remaining≈6m 32s  (ETA ~ 2025-10-26 18:02:03)


```python
def compute_translation_cost_loop(online: torch.Tensor, geo: torch.Tensor, radius: int):
    pad = (radius, radius, radius, radius)
    geo_padded = F.pad(geo, pad=pad, mode="constant", value=0.0)
    B, C, H, W = online.shape
    cost_list = []
    for b in range(B):
        kernel = online[b:b + 1]  # [1, C, H, W]
        corr = F.conv2d(geo_padded[b:b + 1], kernel, padding=0)
        cost_list.append(corr.squeeze(0))  # [1, 2r+1, 2r+1]
    cost = torch.stack(cost_list, dim=0).squeeze(1)  # (B, 2r+1, 2r+1)
    return cost

# ------------------------
# Optimized (grouped conv) version
# ------------------------
def compute_translation_cost_grouped(online: torch.Tensor, geo: torch.Tensor, radius: int):
    pad = (radius, radius, radius, radius)
    geo_padded = F.pad(geo, pad=pad, mode="constant", value=0.0)
    B, C, H, W = online.shape
    _, _, H_p, W_p = geo_padded.shape

    geo_inp = geo_padded.reshape(1, B * C, H_p, W_p)
    kernel_wt = online  # [B, C, H, W]
    cost = F.conv2d(geo_inp, kernel_wt, padding=0, groups=B)
    return cost.squeeze(0)
```







```python

    # def compute_orientation_cost(self, online: Tensor, geo: Tensor) -> Tensor:
    #     device = online.device
    #     theta_grid = self.theta_grid.to(device)
    #     scores = []
    #     for theta in theta_grid:
    #         rotated = affine_warp(online, angle_deg=float(theta.item()), translate_px=(0.0, 0.0))
    #         score = (rotated * geo).sum(dim=(1, 2, 3))  # cosine-like dot over channels+spatial
    #         scores.append(score)
    #     return torch.stack(scores, dim=-1)  # [B, K_theta]


    def compute_orientation_cost(self, online: Tensor, geo: Tensor) -> Tensor:
        device = online.device
        theta_grid_rad = self.theta_grid.to(device) * (math.pi / 180.0)  # [K]

        B, C, H, W = online.shape
        K = theta_grid_rad.shape[0]
        cos_t = torch.cos(theta_grid_rad)  # [K]
        sin_t = torch.sin(theta_grid_rad)  # [K]

        # [B, K, 2, 3]
        matrices = torch.zeros((B, K, 2, 3), device=device, dtype=online.dtype)
        matrices[:, :, 0, 0] = cos_t.view(1, K)
        matrices[:, :, 0, 1] = -sin_t.view(1, K)
        matrices[:, :, 1, 0] = sin_t.view(1, K)
        matrices[:, :, 1, 1] = cos_t.view(1, K)

        matrices = matrices.view(B * K, 2, 3)  # [B*K, 2, 3]
        grid = F.affine_grid(matrices, [B * K, C, H, W], align_corners=False)

        online_rep = online.repeat_interleave(K, dim=0)  # [B*K, C, H, W]
        geo_rep    = geo.repeat_interleave(K, dim=0)              # [B*K, C, H, W]

        rotated = F.grid_sample(online_rep, grid, align_corners=False, mode='bilinear')
        scores = (rotated * geo_rep).sum(dim=(1, 2, 3))  # [B*K]
        return scores.view(B, K)                          # [B, K]

```



BN, IN, GN,

Here you go—two tidy tables: last-epoch metrics and best validation (by lowest val_loss) across the 30 epochs you ran.

Last epoch (epoch 30)
Norm	train_loss	train_xy (m)	train_th (rad)	val_loss	val_xy (m)	val_th (rad)
BN	2.6475	1.3163	0.0149	3.1141	1.5484	0.0174
IN	2.2482	1.1153	0.0175	2.2526	1.1148	0.0230
GN	2.3936	1.1884	0.0168	2.6504	1.3172	0.0161
Best validation over training (lowest val_loss)
Norm	Best epoch	val_loss (best)	val_xy (m)	val_th (rad)
BN	27	2.1953	1.0840	0.0274
IN	19	2.0109	0.9900	0.0315
GN	29	2.3933	1.1880	0.0173


Method:
 Softmax (Convert to Probabilities)
```python
flat_logits = logits.view(B, -1)  # Flatten: (B, 9×9) = (B, 81)
prob = F.softmax(flat_logits, dim=-1).view(B, H, W)  # (B, 9, 9)
```


400epoch 



Method2:
  Sliding Window Correlation
           ↓
      Logits (9×9 scores)
           ↓
      ┌────┴────┐
      ↓         ↓
  Softmax    Gaussian Fit
  (weighted)  (geometric)
      ↓         ↓
   LOSS    Monitoring Only
      ↓
  Backprop
      ↓
  Model Learns!



# Better resolutoin better result .
[Best] Saved model from epoch 82: val_loss=0.6440  | val_rms_theta=0.4426 deg
       Pixel RMS (m):   0.135 x, 0.135 y
       Softmax RMS (m): 0.092 x, 0.114 y
       Gaussian RMS (m): 0.135 x, 0.135 y


# 3 reoslution data train , each datase show the fillment rate
# Data fillment rate , bigger range.0.3047re 329x329 ,100mx100m will have lower lidar points fillment than the 150x150 , 30x30 0.2re

# Edge case 000VFSWWAAkobywItdrErpC6fedKDWg4_020  05tM3HQakLPqAUqQ7iF7uBoar8V0o8WD_011

# gaussian fit not differentiable , not include into loss only apply infer

# train result softmax.
# infer result gaussian