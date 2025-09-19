"""GUI for browsing Argoverse 2 LiDAR feather files and previewing them with Open3D."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pyarrow.feather as feather
import tkinter as tk
from tkinter import filedialog, messagebox

DEFAULT_ROOT = Path("..") / "argverse_data_preview"
POSE_COLUMNS = ["timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]


@dataclass
class PointCloud:
    points: np.ndarray
    intensity: np.ndarray


@dataclass
class SE3:
    rotation: np.ndarray  # shape (3, 3)
    translation: np.ndarray  # shape (3,)


def load_point_cloud(feather_path: Path, *, stride: int) -> PointCloud:
    table = feather.read_table(feather_path, columns=["x", "y", "z", "intensity"])
    frame = table.to_pandas()
    points = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
    intensity = frame["intensity"].to_numpy(dtype=np.float32)
    if stride > 1:
        points = points[::stride]
        intensity = intensity[::stride]
    return PointCloud(points=points, intensity=intensity)


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return intensity
    finite_mask = np.isfinite(intensity)
    if not np.any(finite_mask):
        return np.zeros_like(intensity)
    finite_values = intensity[finite_mask]
    low = float(finite_values.min())
    high = float(finite_values.max())
    if high - low <= 1e-6:
        scaled = np.zeros_like(intensity)
        scaled[finite_mask] = 0.5
        return scaled
    scaled = np.zeros_like(intensity)
    scaled[finite_mask] = (finite_values - low) / (high - low)
    return scaled


def intensity_to_color(intensity: np.ndarray) -> np.ndarray:
    scaled = normalize_intensity(intensity)
    colors = np.zeros((scaled.size, 3), dtype=np.float32)
    colors[:, 0] = scaled
    colors[:, 1] = scaled
    colors[:, 2] = 1.0 - 0.5 * scaled
    return colors


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero magnitude.")
    qw, qx, qy, qz = (q / norm).tolist()
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def apply_se3(points: np.ndarray, transform: SE3) -> np.ndarray:
    return points @ transform.rotation.T + transform.translation


def visualize(cloud: PointCloud, title: str) -> None:
    if cloud.points.size == 0:
        raise ValueError("No points found in frame.")
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(cloud.points)
    geometry.colors = o3d.utility.Vector3dVector(intensity_to_color(cloud.intensity))
    o3d.visualization.draw_geometries([geometry], window_name=title, width=1280, height=800)


def visualize_many(clouds: Iterable[PointCloud], title: str) -> None:
    point_blocks = []
    color_blocks = []
    for cloud in clouds:
        if cloud.points.size == 0:
            continue
        point_blocks.append(cloud.points)
        color_blocks.append(intensity_to_color(cloud.intensity))
    if not point_blocks:
        raise ValueError("No points found across frames.")
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.vstack(point_blocks))
    geometry.colors = o3d.utility.Vector3dVector(np.vstack(color_blocks))
    o3d.visualization.draw_geometries([geometry], window_name=title, width=1280, height=800)


def find_feather_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.feather") if p.is_file())


class PoseLookup:
    def __init__(self, path: Path) -> None:
        self.path = path
        table = feather.read_table(path, columns=POSE_COLUMNS)
        frame = table.to_pandas()
        self._poses: Dict[int, SE3] = {}
        for row in frame.itertuples(index=False):
            rotation = quaternion_to_rotation_matrix(row.qw, row.qx, row.qy, row.qz)
            translation = np.array([row.tx_m, row.ty_m, row.tz_m], dtype=np.float32)
            self._poses[int(row.timestamp_ns)] = SE3(rotation=rotation, translation=translation)

    def get(self, timestamp: int) -> Optional[SE3]:
        return self._poses.get(timestamp)

    @property
    def count(self) -> int:
        return len(self._poses)


class PreviewApp:
    def __init__(self, *, root: Path, stride: int) -> None:
        self.root_dir = root
        self.stride = stride
        self.files: List[Path] = []

        self.window = tk.Tk()
        self.window.title("Argoverse LiDAR Preview")
        self.window.geometry("860x520")

        self.path_var = tk.StringVar(value=str(root))
        self.pose_path_var = tk.StringVar()
        self.stride_var = tk.IntVar(value=max(1, stride))

        self.pose_lookup: Optional[PoseLookup] = None
        self.sensor_transform: Optional[SE3] = None
        self.sensor_name: Optional[str] = None
        self._no_pose_warning_shown = False

        self._build_layout()
        self.refresh_file_list()

    def _build_layout(self) -> None:
        top_frame = tk.Frame(self.window)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top_frame, text="Preview folder:").pack(side=tk.LEFT)
        entry = tk.Entry(top_frame, textvariable=self.path_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Reload", command=self.refresh_file_list).pack(side=tk.LEFT, padx=(5, 0))

        pose_frame = tk.Frame(self.window)
        pose_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Label(pose_frame, text="Pose file:").pack(side=tk.LEFT)
        pose_entry = tk.Entry(pose_frame, textvariable=self.pose_path_var)
        pose_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(pose_frame, text="Browse", command=self.browse_pose).pack(side=tk.LEFT)
        tk.Button(pose_frame, text="Clear", command=self.clear_pose).pack(side=tk.LEFT, padx=(5, 0))

        middle_frame = tk.Frame(self.window)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        scrollbar = tk.Scrollbar(middle_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(middle_frame, activestyle="dotbox")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        control_frame = tk.Frame(self.window)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Label(control_frame, text="Stride:").pack(side=tk.LEFT)
        stride_spin = tk.Spinbox(control_frame, from_=1, to=50, textvariable=self.stride_var, width=5)
        stride_spin.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(control_frame, text="Open Selected", command=self.open_selected).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Open All", command=self.open_all).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(control_frame, text="Open Random", command=self.open_random).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(control_frame, text="Quit", command=self.window.destroy).pack(side=tk.RIGHT)
    def browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.path_var.get() or str(DEFAULT_ROOT))
        if chosen:
            self.path_var.set(chosen)
            self.refresh_file_list()

    def browse_pose(self) -> None:
        initial = self.pose_path_var.get() or str(self.root_dir)
        chosen = filedialog.askopenfilename(
            initialdir=initial,
            title="Select city_SE3_egovehicle.feather",
            filetypes=[("Feather files", "*.feather"), ("All files", "*.*")],
        )
        if chosen:
            self._set_pose_lookup(Path(chosen))

    def clear_pose(self, silent: bool = False) -> None:
        self.pose_lookup = None
        self.pose_path_var.set("")
        self._no_pose_warning_shown = False
        if not silent:
            messagebox.showinfo("Pose cleared", "Pose file cleared; sweeps will render in sensor frame.")

    def refresh_file_list(self) -> None:
        try:
            root_path = Path(self.path_var.get()).expanduser().resolve()
        except Exception as exc:
            messagebox.showerror("Invalid path", f"Could not resolve folder: {exc}")
            return

        self.root_dir = root_path
        self._no_pose_warning_shown = False

        files = find_feather_files(root_path)
        self.files = files
        self.listbox.delete(0, tk.END)
        for file in files:
            try:
                relative = file.relative_to(root_path)
            except ValueError:
                relative = file
            self.listbox.insert(tk.END, str(relative))
        if not files:
            self.listbox.insert(tk.END, "(No .feather files found)")

        self._load_sensor_transform()
        self._maybe_autoload_pose(root_path)

    def _load_sensor_transform(self) -> None:
        self.sensor_transform = None
        self.sensor_name = None
        calib_path = self.root_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        if not calib_path.exists():
            return
        try:
            table = feather.read_table(calib_path)
            frame = table.to_pandas()
        except Exception as exc:
            messagebox.showwarning("Calibration error", f"Failed to load sensor calibration: {exc}")
            return
        if frame.empty:
            return
        candidate = None
        if "sensor_name" in frame.columns:
            lidar_rows = frame[frame["sensor_name"].str.contains("lidar", case=False, na=False)]
            if not lidar_rows.empty:
                up_rows = lidar_rows[lidar_rows["sensor_name"].str.contains("up", case=False, na=False)]
                if not up_rows.empty:
                    candidate = up_rows.iloc[0]
                else:
                    candidate = lidar_rows.iloc[0]
        if candidate is None:
            candidate = frame.iloc[0]
        rotation = quaternion_to_rotation_matrix(candidate.qw, candidate.qx, candidate.qy, candidate.qz)
        translation = np.array([candidate.tx_m, candidate.ty_m, candidate.tz_m], dtype=np.float32)
        self.sensor_transform = SE3(rotation=rotation, translation=translation)
        self.sensor_name = candidate.get("sensor_name", None)

    def _maybe_autoload_pose(self, root_path: Path) -> None:
        default_pose = root_path / "city_SE3_egovehicle.feather"
        current_text = self.pose_path_var.get()
        current_path = None
        if current_text:
            try:
                current_path = Path(current_text).expanduser().resolve()
            except Exception:
                current_path = None
        if default_pose.exists():
            if current_path is None or current_path != default_pose.resolve():
                self._set_pose_lookup(default_pose)
        else:
            if current_path and not current_path.exists():
                self.clear_pose(silent=True)

    def _set_pose_lookup(self, path: Path) -> None:
        try:
            lookup = PoseLookup(path)
        except Exception as exc:
            messagebox.showerror("Pose load error", f"Failed to load pose feather: {exc}")
            return
        if lookup.count == 0:
            messagebox.showwarning("Empty pose file", "Selected pose file contains no rows.")
            return
        self.pose_lookup = lookup
        self.pose_path_var.set(str(path))
        self._no_pose_warning_shown = False
        print(f"Loaded {lookup.count} pose entries from {path}")

    def open_selected(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "No feather files to open.")
            return
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a feather file first.")
            return
        index = selection[0]
        file_path = self.files[index]
        self._open_file(file_path, require_pose=False)

    def open_all(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "No feather files to open.")
            return
        stride = max(1, self.stride_var.get())
        require_pose = self.pose_lookup is not None
        if not require_pose and not self._no_pose_warning_shown:
            messagebox.showinfo(
                "No pose file",
                "No pose feather selected. Sweeps will render in the sensor frame and overlap at the origin.",
            )
            self._no_pose_warning_shown = True
        clouds: List[PointCloud] = []
        skipped: List[str] = []
        for path in self.files:
            try:
                timestamp = self._timestamp_from_path(path)
            except ValueError as exc:
                skipped.append(f"{path.name}: {exc}")
                continue
            try:
                cloud = load_point_cloud(path, stride=stride)
            except Exception as exc:
                messagebox.showerror("Failed to open feather", f"{path}: {exc}")
                return
            try:
                cloud, missing_pose = self._transform_cloud(cloud, timestamp, require_pose=require_pose)
            except ValueError as exc:
                skipped.append(f"{path.name}: {exc}")
                continue
            if missing_pose:
                skipped.append(f"{path.name}: pose not found")
                continue
            clouds.append(cloud)
        if not clouds:
            detail = "\n".join(skipped[:5]) if skipped else ""
            if detail:
                messagebox.showerror("Nothing to display", f"No sweeps were rendered.\n{detail}")
            else:
                messagebox.showinfo("Nothing to display", "No sweeps satisfied the filters.")
            return
        if skipped:
            summary = f"Skipped {len(skipped)} sweep(s) without pose matches."
            if len(skipped) <= 5:
                summary += "\n" + "\n".join(skipped)
            messagebox.showwarning("Missing poses", summary)
        total_points = sum(cloud.points.shape[0] for cloud in clouds)
        sensor_note = f" using {self.sensor_name}" if self.sensor_name else ""
        print(
            f"Loaded {total_points} points from {len(clouds)} files (stride={stride}){sensor_note}"
        )
        try:
            visualize_many(clouds, f"All sweeps ({len(clouds)})")
        except Exception as exc:
            messagebox.showerror("Visualisation error", f"{exc}")

    def open_random(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "No feather files to choose from.")
            return
        file_path = random.choice(self.files)
        self.listbox.selection_clear(0, tk.END)
        idx = self.files.index(file_path)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self._open_file(file_path, require_pose=False)

    def _open_file(self, path: Path, *, require_pose: bool) -> None:
        stride = max(1, self.stride_var.get())
        try:
            timestamp = self._timestamp_from_path(path)
        except ValueError as exc:
            messagebox.showerror("Invalid filename", str(exc))
            return
        try:
            cloud = load_poin
            cloud = load_point_cloud(path, stride=stride)
        except Exception as exc:
            messagebox.showerror("Failed to open feather", f"{exc}")
            return
        try:
            cloud, missing_pose = self._transform_cloud(cloud, timestamp, require_pose=require_pose)
        except ValueError as exc:
            messagebox.showerror("Pose lookup failed", f"{exc}")
            return
        if missing_pose and self.pose_lookup is not None:
            messagebox.showwarning(
                "Missing pose",
                "Pose entry not found for this timestamp; rendering in the sensor frame.",
            )
        print(f"Loaded {cloud.points.shape[0]} points from {path}")
        try:
            visualize(cloud, path.name)
        except Exception as exc:
            messagebox.showerror("Visualisation error", f"{exc}")

    def _transform_cloud(
        self, cloud: PointCloud, timestamp: int, *, require_pose: bool
    ) -> Tuple[PointCloud, bool]:
        transforms: List[SE3] = []
        if self.sensor_transform is not None:
            transforms.append(self.sensor_transform)
        missing_pose = False
        if self.pose_lookup is not None:
            pose = self.pose_lookup.get(timestamp)
            if pose is None:
                if require_pose:
                    raise ValueError(f"pose not found for timestamp {timestamp}")
                missing_pose = True
            else:
                transforms.append(pose)
        if not transforms:
            return cloud, missing_pose
        points = cloud.points
        for transform in transforms:
            points = apply_se3(points, transform)
        return PointCloud(points=points, intensity=cloud.intensity), missing_pose

    @staticmethod
    def _timestamp_from_path(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError as exc:
            raise ValueError(f"filename does not start with a timestamp: {path.name}") from exc

    def run(self) -> None:
        self.window.mainloop()

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse and preview Argoverse 2 LiDAR sweeps.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"Folder to search for .feather files (default: {DEFAULT_ROOT})")
    parser.add_argument("--stride", type=int, default=1,
                        help="Decimate the point cloud by keeping one of every N points (default: %(default)s).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    app = PreviewApp(root=args.root.resolve(), stride=args.stride)
    app.run()


if __name__ == "__main__":
    main()
