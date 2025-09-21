"""Tkinter GUI to inspect up to two GeoTIFFs side by side with zoomable previews."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


DEFAULT_TIF = Path(__file__).resolve().parents[1] / "Data-Sample" / "lidar_height_intensity.tif"
PREVIEW_BASE_SIZE = 700  # default size of the preview before zooming
ZOOM_STEP = 1.25
ZOOM_MIN = 0.25
ZOOM_MAX = 5.0
TEXT_FRAME_HEIGHT = 220  # keep metadata panel under one-third of default window height


def describe_band(array: np.ndarray, nodata: float | None) -> dict[str, float]:
    data = array.astype(np.float64).reshape(-1)
    if nodata is not None:
        data = data[~np.isclose(data, nodata)]
    data = data[~np.isnan(data)]
    if data.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "p5": float("nan"), "p95": float("nan")}
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "p5": float(np.percentile(data, 5)),
        "p95": float(np.percentile(data, 95)),
    }


def _normalize_for_preview(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = np.nanmin(finite)
        high = np.nanmax(finite)
    if not np.isfinite(high) or high <= low:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - low) / (high - low)
    scaled = np.clip(scaled, 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255).astype(np.uint8)


def build_preview(src: rasterio.io.DatasetReader) -> Image.Image:
    scale = max(src.width, src.height) / PREVIEW_BASE_SIZE
    if scale <= 1:
        out_w, out_h = src.width, src.height
    else:
        out_w = max(1, int(src.width / scale))
        out_h = max(1, int(src.height / scale))

    mask = src.dataset_mask(out_shape=(out_h, out_w))
    has_mask = mask.any()

    if src.count >= 3:
        bands = [1, 2, 3]
        data = src.read(bands, out_shape=(len(bands), out_h, out_w), resampling=Resampling.bilinear)
        nodata = src.nodata
        rgb: List[np.ndarray] = []
        for idx in range(len(bands)):
            arr = data[idx]
            if nodata is not None:
                arr = np.where(np.isclose(arr, nodata), np.nan, arr)
            rgb.append(_normalize_for_preview(arr))
        preview = np.stack(rgb, axis=-1)
    else:
        arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
        if src.nodata is not None:
            arr = np.where(np.isclose(arr, src.nodata), np.nan, arr)
        gray = _normalize_for_preview(arr)
        preview = np.stack([gray] * 3, axis=-1)

    if has_mask:
        valid = mask > 0
        preview[~valid] = 0

    return Image.fromarray(preview)


def analyze_raster(path: Path) -> Tuple[str, Image.Image]:
    with rasterio.open(path) as src:
        info_lines = [
            f"File: {path}",
            f"Driver: {src.driver}",
            f"Size (width x height): {src.width} x {src.height}",
            f"Band count: {src.count}",
            f"CRS: {src.crs}",
            f"Transform: {src.transform}",
            f"Resolution: {src.res}",
            f"Nodata: {src.nodata}",
        ]
        mask = src.dataset_mask()
        coverage = float(mask.astype(bool).sum()) / mask.size * 100.0
        info_lines.append(f"Valid coverage: {coverage:.2f}%")

        for band_idx in range(1, src.count + 1):
            band = src.read(band_idx)
            stats = describe_band(band, src.nodata)
            info_lines.append(
                "Band {idx} stats: min={min:.3f}, max={max:.3f}, mean={mean:.3f}, p5={p5:.3f}, p95={p95:.3f}".format(
                    idx=band_idx,
                    **stats,
                )
            )

        preview = build_preview(src)

    return "\n".join(info_lines), preview


class RasterPanel:
    def __init__(self, parent: tk.Widget, title: str) -> None:
        self.frame = tk.Frame(parent, borderwidth=1, relief=tk.GROOVE)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(self.frame, text=title, font=("Segoe UI", 12, "bold"))
        title_label.pack(anchor="w", pady=(0, 4))

        controls = tk.Frame(self.frame)
        controls.pack(fill=tk.X, pady=(0, 6))

        self.path_var = tk.StringVar(value="No file loaded")
        open_button = tk.Button(controls, text="Open GeoTIFF", command=self.choose_file)
        open_button.pack(side=tk.LEFT)

        path_label = tk.Label(controls, textvariable=self.path_var, anchor="w")
        path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        text_frame = tk.Frame(self.frame, height=TEXT_FRAME_HEIGHT)
        text_frame.pack(fill=tk.X)
        text_frame.pack_propagate(False)
        self.text = ScrolledText(text_frame, width=60)
        self.text.pack(fill=tk.BOTH, expand=True)

        preview_container = tk.Frame(self.frame)
        preview_container.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        preview_controls = tk.Frame(preview_container)
        preview_controls.pack(fill=tk.X, pady=(0, 4))

        zoom_out_btn = tk.Button(preview_controls, text="-", width=3, command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT)

        zoom_in_btn = tk.Button(preview_controls, text="+", width=3, command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=(6, 0))

        reset_btn = tk.Button(preview_controls, text="Reset", command=self.reset_zoom)
        reset_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.zoom_label_var = tk.StringVar(value="Zoom: 100%")
        zoom_label = tk.Label(preview_controls, textvariable=self.zoom_label_var)
        zoom_label.pack(side=tk.LEFT, padx=(8, 0))

        self.preview_canvas = tk.Canvas(preview_container, bg="#222")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self.photo: ImageTk.PhotoImage | None = None
        self.preview_image: Image.Image | None = None
        self.zoom_factor: float = 1.0

        self.preview_canvas.bind("<Configure>", lambda _event: self.render_preview())

    def choose_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if file_path:
            self.load(Path(file_path))

    def load(self, path: Path) -> None:
        try:
            metadata_text, preview_image = analyze_raster(path)
        except Exception as exc:
            messagebox.showerror("Failed to load raster", str(exc))
            return
        self.path_var.set(str(path))
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, metadata_text)

        self.preview_image = preview_image
        self.zoom_factor = 1.0
        self.update_zoom_label()
        self.render_preview()

    def render_preview(self) -> None:
        self.preview_canvas.delete("all")
        if self.preview_image is None:
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() / 2,
                self.preview_canvas.winfo_height() / 2,
                fill="white",
                text="Preview",
            )
            return

        width = int(self.preview_image.width * self.zoom_factor)
        height = int(self.preview_image.height * self.zoom_factor)
        width = max(1, width)
        height = max(1, height)

        resized = self.preview_image.resize((width, height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        x = max((canvas_width - width) / 2, 0)
        y = max((canvas_height - height) / 2, 0)
        self.preview_canvas.create_image(x, y, anchor="nw", image=self.photo)

    def zoom_in(self) -> None:
        if self.preview_image is None:
            return
        self.zoom_factor = min(self.zoom_factor * ZOOM_STEP, ZOOM_MAX)
        self.update_zoom_label()
        self.render_preview()

    def zoom_out(self) -> None:
        if self.preview_image is None:
            return
        self.zoom_factor = max(self.zoom_factor / ZOOM_STEP, ZOOM_MIN)
        self.update_zoom_label()
        self.render_preview()

    def reset_zoom(self) -> None:
        if self.preview_image is None:
            return
        self.zoom_factor = 1.0
        self.update_zoom_label()
        self.render_preview()

    def update_zoom_label(self) -> None:
        percent = int(self.zoom_factor * 100)
        self.zoom_label_var.set(f"Zoom: {percent}%")


class RasterReviewApp:
    def __init__(self, root: tk.Tk, initial_paths: Tuple[Path | None, Path | None]) -> None:
        self.root = root
        self.root.title("GeoTIFF Side-by-Side Reviewer")
        self.root.geometry("1500x900")

        container = tk.Frame(root)
        container.pack(fill=tk.BOTH, expand=True)

        self.left_panel = RasterPanel(container, "Raster A")
        self.right_panel = RasterPanel(container, "Raster B")

        left_path, right_path = initial_paths
        if left_path and left_path.exists():
            self.left_panel.load(left_path)
        if right_path and right_path.exists():
            self.right_panel.load(right_path)


def parse_args() -> Tuple[Path | None, Path | None]:
    parser = argparse.ArgumentParser(description="Open the dual GeoTIFF review GUI.")
    parser.add_argument("paths", nargs="*", type=Path, help="Up to two GeoTIFF files to open (left then right)")
    args = parser.parse_args()

    if len(args.paths) > 2:
        parser.error("Provide at most two GeoTIFF paths: left then right")

    left = args.paths[0] if len(args.paths) >= 1 else None
    right = args.paths[1] if len(args.paths) == 2 else None

    if left is None and DEFAULT_TIF.exists():
        left = DEFAULT_TIF

    return left, right


def main() -> None:
    initial_paths = parse_args()
    root = tk.Tk()
    RasterReviewApp(root, initial_paths)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
