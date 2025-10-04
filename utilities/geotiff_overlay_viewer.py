"""
GeoTIFF + Point Cloud Overlay Viewer

Projects point cloud data onto a GeoTIFF image using UTM coordinates.
Point colors represent elevation (Z values): darker = lower, lighter = higher.
"""

import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_geotiff(path: Path) -> Optional[Tuple[np.ndarray, object, dict]]:
    """Load GeoTIFF file and extract image, transform, and metadata.

    Args:
        path: Path to GeoTIFF file

    Returns:
        Tuple of (image array, transform, metadata) or None if failed
    """
    if not HAS_RASTERIO:
        print("Error: rasterio not installed. Install with: pip install rasterio")
        return None

    try:
        with rasterio.open(path) as src:
            # Read all bands
            image = src.read()

            # Convert to (height, width, bands) format
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))

            # Handle different channel counts
            if image.ndim == 2:
                # Single band grayscale - convert to RGB
                image = np.stack([image, image, image], axis=-1)
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    # Single channel - convert to RGB
                    image = np.stack([image[:, :, 0], image[:, :, 0], image[:, :, 0]], axis=-1)
                elif image.shape[2] == 4:
                    # RGBA - take only RGB channels
                    image = image[:, :, :3]
                elif image.shape[2] > 4:
                    # More than 4 channels - take first 3
                    image = image[:, :, :3]
                # If shape[2] == 3, it's already RGB, no change needed

            # Normalize to 0-255 if needed
            if image.dtype != np.uint8:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            transform = src.transform
            metadata = {
                'crs': src.crs,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
            }

            print(f"Loaded GeoTIFF: {path.name}")
            print(f"  Size: {metadata['width']} x {metadata['height']} pixels")
            print(f"  Bounds: {metadata['bounds']}")
            print(f"  CRS: {metadata['crs']}")

            return image, transform, metadata

    except Exception as e:
        print(f"Error loading GeoTIFF {path}: {e}")
        return None


def load_parquet_points(path: Path) -> Optional[np.ndarray]:
    """Load point cloud from parquet file.

    Returns:
        Nx3 array (E, N, Z) or None if failed
    """
    try:
        df = pd.read_parquet(path)

        # Try to find XYZ columns
        if all(col in df.columns for col in ["utm_e", "utm_n", "elevation"]):
            points = df[["utm_e", "utm_n", "elevation"]].to_numpy(dtype=float)
        elif all(col in df.columns for col in ["utm_e", "utm_n", "z"]):
            points = df[["utm_e", "utm_n", "z"]].to_numpy(dtype=float)
        elif all(col in df.columns for col in ["x", "y", "z"]):
            points = df[["x", "y", "z"]].to_numpy(dtype=float)
        else:
            print(f"Warning: Could not find XYZ columns in {path}")
            return None

        print(f"Loaded {len(points):,} points from {path}")
        print(f"  E range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  N range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        return points

    except Exception as e:
        print(f"Error loading parquet {path}: {e}")
        return None


def project_points_to_image(
    points: np.ndarray,
    transform: object,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project UTM points to image pixel coordinates.

    Args:
        points: Nx3 array (E, N, Z)
        transform: Rasterio affine transform
        image_shape: (height, width) of image

    Returns:
        Tuple of (valid_rows, valid_cols, valid_z_values)
    """
    # Convert UTM to pixel coordinates
    rows, cols = rowcol(transform, points[:, 0], points[:, 1])
    rows = np.array(rows)
    cols = np.array(cols)

    # Filter points within image bounds
    height, width = image_shape[:2]
    valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_z = points[valid_mask, 2]

    print(f"Projected {len(valid_rows):,} / {len(points):,} points within image bounds")

    return valid_rows, valid_cols, valid_z


def create_overlay_image(
    base_image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    z_values: np.ndarray,
    point_size: int = 2,
    alpha: float = 0.7,
) -> np.ndarray:
    """Create overlay image with points colored by elevation.

    Args:
        base_image: Base GeoTIFF image (H, W, 3)
        rows: Point row coordinates
        cols: Point column coordinates
        z_values: Point Z values (elevation)
        point_size: Size of each point in pixels
        alpha: Transparency of points (0=transparent, 1=opaque)

    Returns:
        Overlay image with points
    """
    # Create copy of base image as float for better blending
    overlay = base_image.copy().astype(np.float32)

    # Create an accumulation layer to show overlap
    # This layer accumulates transparency - more overlaps = more opaque
    alpha_accum = np.zeros(base_image.shape[:2], dtype=np.float32)

    # Normalize Z values to 0-1 range
    z_min, z_max = z_values.min(), z_values.max()
    if z_max > z_min:
        z_norm = (z_values - z_min) / (z_max - z_min)
    else:
        z_norm = np.ones_like(z_values) * 0.5

    # Create colormap: darker (blue) for low elevation, lighter (yellow) for high elevation
    # Using a custom colormap from blue (low) to yellow (high)
    try:
        # New matplotlib API (3.7+)
        cmap = plt.colormaps['viridis']
    except AttributeError:
        # Older matplotlib API
        cmap = plt.cm.get_cmap('viridis')

    # Draw points on overlay with proper transparency accumulation
    height, width = overlay.shape[:2]

    for row, col, z in zip(rows, cols, z_norm):
        # Get color from colormap
        color = cmap(z)[:3]  # RGB values (0-1)
        color_255 = np.array(color) * 255.0  # Keep as float

        # Draw point with given size
        r_min = max(0, int(row) - point_size // 2)
        r_max = min(height, int(row) + point_size // 2 + 1)
        c_min = max(0, int(col) - point_size // 2)
        c_max = min(width, int(col) + point_size // 2 + 1)

        # Get the region to update
        region = overlay[r_min:r_max, c_min:c_max]
        alpha_region = alpha_accum[r_min:r_max, c_min:c_max]

        # Calculate current alpha for this region (accounts for existing transparency)
        # The more points overlap, the more visible they become
        current_alpha = 1.0 - (1.0 - alpha) * (1.0 - alpha_region)

        # Blend with accumulated alpha
        # This creates a "layered transparency" effect where overlaps are more visible
        blended = (alpha * color_255[np.newaxis, np.newaxis, :] +
                   (1 - alpha) * region)

        overlay[r_min:r_max, c_min:c_max] = blended

        # Update alpha accumulation
        alpha_accum[r_min:r_max, c_min:c_max] = current_alpha

    # Convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    print(f"Created overlay with {len(rows):,} points")
    print(f"  Z range: [{z_min:.2f}, {z_max:.2f}]")
    print(f"  Point size: {point_size} pixels")
    print(f"  Alpha: {alpha}")
    print(f"  Max overlap accumulation: {alpha_accum.max():.2f}")

    return overlay


def upsample_image(image: np.ndarray, scale_factor: int = 10) -> np.ndarray:
    """Upsample image to higher resolution.

    Args:
        image: Input image (H, W, 3)
        scale_factor: Upsampling factor (default: 10x)

    Returns:
        Upsampled image (H*scale, W*scale, 3)
    """
    from PIL import Image

    # Convert to PIL Image
    pil_img = Image.fromarray(image)

    # Calculate new size
    new_width = pil_img.width * scale_factor
    new_height = pil_img.height * scale_factor

    # Upsample using high-quality Lanczos resampling
    upsampled = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    print(f"Upsampled image from {pil_img.width}x{pil_img.height} to {new_width}x{new_height} ({scale_factor}x)")

    return np.array(upsampled)


class OverlayViewerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GeoTIFF + Point Cloud Overlay Viewer")

        # File paths
        self.geotiff_path = tk.StringVar(value="")
        self.points_path = tk.StringVar(value="")

        # Parameters
        self.point_size_var = tk.IntVar(value=2)
        self.alpha_var = tk.DoubleVar(value=0.3)  # More transparent to show overlaps

        # Data
        self.overlay_image = None
        self.figure = None
        self.canvas = None

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="GeoTIFF + Point Cloud Overlay Viewer", font=("", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)

        # GeoTIFF file
        geotiff_frame = ttk.LabelFrame(main_frame, text="GeoTIFF File", padding="10")
        geotiff_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Entry(geotiff_frame, textvariable=self.geotiff_path, width=60).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(geotiff_frame, text="Browse", command=self.browse_geotiff).grid(row=0, column=1, padx=5, pady=5)

        # Points file
        points_frame = ttk.LabelFrame(main_frame, text="Point Cloud File (Parquet)", padding="10")
        points_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Entry(points_frame, textvariable=self.points_path, width=60).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(points_frame, text="Browse", command=self.browse_points).grid(row=0, column=1, padx=5, pady=5)

        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Overlay Parameters", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(params_frame, text="Point Size (pixels):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        ttk.Entry(params_frame, textvariable=self.point_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(params_frame, text="Alpha (transparency):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        ttk.Entry(params_frame, textvariable=self.alpha_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)

        ttk.Label(params_frame, text="(0.0=transparent, 1.0=opaque, lower = see overlaps)", font=("", 8)).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Generate Overlay", command=self.generate_overlay, width=20).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Save JPEG", command=self.save_jpeg, width=20).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit, width=20).grid(row=0, column=2, padx=5)

        # Info
        info_frame = ttk.LabelFrame(main_frame, text="Color Legend", padding="10")
        info_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        info_text = "Point colors represent elevation (Z values):\nBlue (dark) = Low elevation → Yellow/Green (light) = High elevation"
        ttk.Label(info_frame, text=info_text, font=("", 9)).grid(row=0, column=0)

    def browse_geotiff(self):
        """Browse for GeoTIFF file."""
        path = filedialog.askopenfilename(
            title="Select GeoTIFF File",
            filetypes=[("GeoTIFF Files", "*.tif *.tiff"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            self.geotiff_path.set(path)

    def browse_points(self):
        """Browse for parquet file."""
        path = filedialog.askopenfilename(
            title="Select Point Cloud Parquet File",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            self.points_path.set(path)

    def generate_overlay(self):
        """Generate overlay image."""
        try:
            # Validate inputs
            geotiff_path = Path(self.geotiff_path.get().strip())
            points_path = Path(self.points_path.get().strip())

            if not geotiff_path.is_file():
                messagebox.showerror("Error", f"GeoTIFF file not found:\n{geotiff_path}")
                return

            if not points_path.is_file():
                messagebox.showerror("Error", f"Points file not found:\n{points_path}")
                return

            # Load data
            print("\n=== Loading Data ===")
            geotiff_data = load_geotiff(geotiff_path)
            if geotiff_data is None:
                messagebox.showerror("Error", "Failed to load GeoTIFF file")
                return

            base_image, transform, metadata = geotiff_data

            points = load_parquet_points(points_path)
            if points is None:
                messagebox.showerror("Error", "Failed to load points file")
                return

            # Project points to image
            print("\n=== Projecting Points ===")
            rows, cols, z_values = project_points_to_image(points, transform, base_image.shape)

            if len(rows) == 0:
                messagebox.showwarning("Warning", "No points projected within image bounds. Check coordinate systems.")
                return

            # Create overlay
            print("\n=== Creating Overlay ===")
            point_size = self.point_size_var.get()
            alpha = self.alpha_var.get()

            self.overlay_image = create_overlay_image(
                base_image,
                rows,
                cols,
                z_values,
                point_size=point_size,
                alpha=alpha,
            )

            # Display in matplotlib
            self.display_overlay()

            messagebox.showinfo("Success", f"Overlay generated successfully!\n\n{len(rows):,} points projected onto image.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate overlay:\n{str(e)}")
            print(f"Error: {e}")

    def display_overlay(self):
        """Display overlay image in GUI."""
        if self.overlay_image is None:
            return

        # Create new window for display
        display_window = tk.Toplevel(self.root)
        display_window.title("Overlay Result")

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.overlay_image)
        ax.set_title("GeoTIFF + Point Cloud Overlay\n(Colors: Blue=Low elevation, Yellow=High elevation)")
        ax.axis('off')

        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=display_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Store references
        self.figure = fig
        self.canvas = canvas

    def save_jpeg(self):
        """Save overlay image as JPEG with 10x upsampling."""
        if self.overlay_image is None:
            messagebox.showwarning("Warning", "Please generate overlay first")
            return

        try:
            # Ask for save location
            save_path = filedialog.asksaveasfilename(
                title="Save Overlay Image (10x resolution)",
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")],
                initialdir=Path.cwd(),
                initialfile="overlay_result_10x.jpg",
            )

            if not save_path:
                return

            print(f"\n=== Upsampling and Saving Overlay ===")

            # Upsample to 10x resolution
            upsampled = upsample_image(self.overlay_image, scale_factor=10)

            # Save using PIL
            img = Image.fromarray(upsampled)
            img.save(save_path, quality=95)

            print(f"Saved to: {save_path}")

            messagebox.showinfo("Success", f"Overlay saved (10x upsampled) to:\n{save_path}\n\nOriginal: {self.overlay_image.shape[1]}x{self.overlay_image.shape[0]}\nSaved: {upsampled.shape[1]}x{upsampled.shape[0]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")


def main():
    parser = argparse.ArgumentParser(description="GeoTIFF + Point Cloud Overlay Viewer")
    parser.add_argument("--geotiff", type=Path, help="Path to GeoTIFF file")
    parser.add_argument("--points", type=Path, help="Path to point cloud parquet file")
    parser.add_argument("--point-size", type=int, default=2, help="Point size in pixels")
    parser.add_argument("--alpha", type=float, default=0.3, help="Point transparency (0-1, default 0.3 for overlap visibility)")
    parser.add_argument("--output", type=Path, help="Output JPEG file path (will be 10x upsampled)")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    args = parser.parse_args()

    # Check for rasterio
    if not HAS_RASTERIO:
        print("Error: rasterio is required but not installed.")
        print("Install with: pip install rasterio")
        return

    # Launch GUI if requested or if no files specified
    if args.gui or not (args.geotiff and args.points):
        root = tk.Tk()
        app = OverlayViewerGUI(root)
        root.mainloop()
        return

    # Command-line mode
    print("=== GeoTIFF + Point Cloud Overlay ===")

    # Load data
    geotiff_data = load_geotiff(args.geotiff)
    if geotiff_data is None:
        return

    base_image, transform, metadata = geotiff_data

    points = load_parquet_points(args.points)
    if points is None:
        return

    # Project points
    rows, cols, z_values = project_points_to_image(points, transform, base_image.shape)

    if len(rows) == 0:
        print("Error: No points projected within image bounds")
        return

    # Create overlay
    overlay_image = create_overlay_image(
        base_image,
        rows,
        cols,
        z_values,
        point_size=args.point_size,
        alpha=args.alpha,
    )

    # Save or display
    if args.output:
        # Upsample to 10x resolution before saving
        print("\n=== Upsampling to 10x Resolution ===")
        upsampled = upsample_image(overlay_image, scale_factor=10)

        img = Image.fromarray(upsampled)
        img.save(args.output, quality=95)
        print(f"\nSaved 10x upsampled overlay to: {args.output}")
        print(f"  Original size: {overlay_image.shape[1]}x{overlay_image.shape[0]}")
        print(f"  Saved size: {upsampled.shape[1]}x{upsampled.shape[0]}")
    else:
        # Display with matplotlib
        plt.figure(figsize=(12, 10))
        plt.imshow(overlay_image)
        plt.title("GeoTIFF + Point Cloud Overlay\n(Colors: Blue=Low, Yellow=High)\n(Preview - save will be 10x resolution)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
