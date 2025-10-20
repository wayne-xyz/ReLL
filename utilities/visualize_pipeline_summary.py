"""Visualize pipeline summary CSV.

Produces distribution plots and scatter/relationship plots for these fields:
- point_count
- duration_s
- sensor_motion_length_m
- sensor_displacement_m
- z_offset_m

Saves PNG files to utilities/plots/.

Usage:
    python utilities/visualize_pipeline_summary.py \
        --csv ../processed_samples_austin_train/pipeline_test_summary.csv \
        --outdir plots

The script uses pandas, matplotlib and seaborn (if available). It falls back to matplotlib-only if seaborn is not installed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

NUMERIC_FIELDS = [
    "point_count",
    "duration_s",
    "sensor_motion_length_m",
    "sensor_displacement_m",
    "z_offset_m",
]


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure numeric types
    for f in NUMERIC_FIELDS:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def plot_distributions(df: pd.DataFrame, outdir: Path) -> None:
    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            continue
        series = df[field].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(8, 5))
        if _HAS_SEABORN:
            sns.histplot(series, kde=True)
        else:
            plt.hist(series, bins=50, alpha=0.7)
        plt.title(f"Distribution of {field}")
        plt.xlabel(field)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / f"dist_{field}.png", dpi=150)
        plt.close()

        # Boxplot
        plt.figure(figsize=(6, 2))
        if _HAS_SEABORN:
            sns.boxplot(x=series)
        else:
            plt.boxplot(series, vert=False)
        plt.title(f"Boxplot of {field}")
        plt.tight_layout()
        plt.savefig(outdir / f"box_{field}.png", dpi=150)
        plt.close()


def plot_pairwise(df: pd.DataFrame, outdir: Path) -> None:
    # Scatter plots paired: z_offset_m vs others, point_count vs duration
    pairs = [
        ("z_offset_m", "point_count"),
        ("z_offset_m", "duration_s"),
        ("z_offset_m", "sensor_motion_length_m"),
        ("z_offset_m", "sensor_displacement_m"),
        ("point_count", "duration_s"),
        ("sensor_motion_length_m", "sensor_displacement_m"),
    ]

    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        sub = df[[x, y]].dropna()
        if sub.empty:
            continue
        plt.figure(figsize=(6, 5))
        if _HAS_SEABORN:
            sns.scatterplot(data=sub, x=x, y=y)
        else:
            plt.scatter(sub[x], sub[y], s=10, alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{y}_vs_{x}.png", dpi=150)
        plt.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True, help="Path to pipeline summary CSV")
    p.add_argument("--outdir", type=Path, default=Path("plots"), help="Output directory (under utilities)")
    args = p.parse_args(argv)

    csv_path = args.csv
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return 2

    outdir = Path(__file__).parent / args.outdir
    ensure_outdir(outdir)

    df = load_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_distributions(df, outdir)
    plot_pairwise(df, outdir)

    # Summary figure: simple table of statistics
    stats = df[NUMERIC_FIELDS].describe().T
    stats_file = outdir / "summary_stats.csv"
    stats.to_csv(stats_file)
    print(f"Wrote summary stats to: {stats_file}")

    print(f"All plots saved to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
