
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import open3d as o3d
from open3d.visualization import gui, rendering

WINDOW_TITLE = "GICP Triple Cloud Viewer"
POINT_SIZE = 2.0

COLORS: Dict[str, tuple] = {
    "target": (0.1, 0.4, 0.9),   # blue-ish
    "source": (0.9, 0.2, 0.2),   # red
    "aligned": (0.2, 0.8, 0.3),  # green
}

LABELS: Dict[str, str] = {
    "target": "Target / Reference",
    "source": "Perturbed Source",
    "aligned": "GICP Aligned",
}

# Configure the files you want to visualise. Use None to leave a slot empty.
FILE_PATHS: Dict[str, Optional[Path]] = {
    "target": "G:/GithubProject/ReLL/gicp-veri/outputs/target_reference.parquet",
    "source": "G:\GithubProject\ReLL\gicp-veri\outputs\source_offset.parquet",
    "aligned": "G:\GithubProject\ReLL\gicp-veri\outputs\source_aligned.parquet",
}


def infer_xyz_columns(df: pd.DataFrame) -> np.ndarray:
    candidates = [
        ("utm_e", "utm_n", "elevation"),
        ("utm_e", "utm_n", "z"),
        ("x", "y", "z"),
    ]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return df.loc[:, cols].to_numpy(dtype=float)
    raise ValueError(
        "Could not infer XYZ columns. Expected one of (utm_e, utm_n, elevation), (utm_e, utm_n, z), or (x, y, z)."
    )


class CloudViewer:
    def __init__(self) -> None:
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(WINDOW_TITLE, 1280, 820)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.02, 0.02, 0.02, 1.0])
        self.scene_widget.enable_scene_caching(True)

        self.window.add_child(self.scene_widget)

        self.panel = gui.Vert(8.0, gui.Margins(12, 12, 12, 12))
        self.window.add_child(self.panel)

        def on_layout(_: gui.LayoutContext) -> None:
            content_rect = self.window.content_rect
            panel_width = 360
            self.scene_widget.frame = gui.Rect(
                content_rect.x,
                content_rect.y,
                content_rect.width - panel_width,
                content_rect.height,
            )
            self.panel.frame = gui.Rect(
                content_rect.get_right() - panel_width,
                content_rect.y,
                panel_width,
                content_rect.height,
            )

        self.window.set_on_layout(on_layout)

        self.file_labels: Dict[str, gui.Label] = {}
        self.checkboxes: Dict[str, gui.Checkbox] = {}
        self.materials: Dict[str, rendering.MaterialRecord] = {}
        self.geometries: Dict[str, o3d.geometry.PointCloud] = {}

        self._build_panel()
        self._autoload_clouds()

    def _build_panel(self) -> None:
        legend = gui.Label("Legend: Blue=Target, Red=Source, Green=Aligned")
        legend.text_color = gui.Color(0.8, 0.8, 0.8)
        self.panel.add_child(legend)

        for key in ("target", "source", "aligned"):
            row = gui.Horiz(8, gui.Margins(0, 0, 0, 0))
            load_button = gui.Button(f"Reload {LABELS[key]}")
            load_button.background_color = gui.Color(*COLORS[key], 1.0)
            load_button.set_on_clicked(lambda k=key: self._load_from_config(k))

            checkbox = gui.Checkbox(f"Show {LABELS[key]}")
            checkbox.checked = False
            checkbox.set_on_checked(lambda checked, k=key: self._toggle_cloud(k, checked))

            self.checkboxes[key] = checkbox

            row.add_child(load_button)
            row.add_child(checkbox)
            self.panel.add_child(row)

            label = gui.Label(self._path_text(key))
            label.text_color = gui.Color(0.7, 0.7, 0.7)
            self.panel.add_child(label)
            self.file_labels[key] = label

        self.panel.add_child(gui.Label(""))
        reset_btn = gui.Button("Reset Camera")
        reset_btn.set_on_clicked(self._reset_camera)
        self.panel.add_child(reset_btn)

        instr = gui.Label("Tip: edit FILE_PATHS in cloud_viewer.py and press Reload to refresh.")
        instr.text_color = gui.Color(0.6, 0.6, 0.6)
        self.panel.add_child(instr)

    def _path_text(self, key: str) -> str:
        path = FILE_PATHS.get(key)
        return str(path) if path else "Path not set"

    def _autoload_clouds(self) -> None:
        print("-- Auto loading configured clouds --")
        for key in FILE_PATHS:
            self._load_from_config(key)

    def _load_from_config(self, key: str) -> None:
        path = FILE_PATHS.get(key)
        if path is None:
            print(f"[{key}] no path configured; skipping.")
            self.file_labels[key].text = "Path not set"
            self.checkboxes[key].checked = False
            if key in self.geometries:
                self.scene_widget.scene.remove_geometry(key)
                del self.geometries[key]
            return

        file_path = Path(path)
        print(f"[{key}] loading: {file_path}")
        if not file_path.exists():
            msg = f"Missing file: {file_path}"
            print(f"[{key}] ERROR {msg}")
            self.file_labels[key].text = msg
            self.checkboxes[key].checked = False
            if key in self.geometries:
                self.scene_widget.scene.remove_geometry(key)
                del self.geometries[key]
            return

        try:
            df = pd.read_parquet(file_path)
            points = infer_xyz_columns(df)
        except Exception as exc:
            msg = f"Failed to load {file_path}: {exc}"
            print(f"[{key}] ERROR {msg}")
            self.file_labels[key].text = msg
            self.checkboxes[key].checked = False
            if key in self.geometries:
                self.scene_widget.scene.remove_geometry(key)
                del self.geometries[key]
            return

        print(f"[{key}] loaded {points.shape[0]} points; columns={list(df.columns)}")

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = POINT_SIZE
        material.base_color = (*COLORS[key], 1.0)

        if key in self.geometries:
            self.scene_widget.scene.remove_geometry(key)

        self.scene_widget.scene.add_geometry(key, pc, material)
        self.materials[key] = material
        self.geometries[key] = pc
        self.file_labels[key].text = str(file_path)

        self.checkboxes[key].checked = True
        self.scene_widget.scene.show_geometry(key, True)
        self._update_camera()

    def _toggle_cloud(self, key: str, checked: bool) -> None:
        if key not in self.geometries:
            return
        self.scene_widget.scene.show_geometry(key, checked)
        self._update_camera()

    def _reset_camera(self) -> None:
        self._update_camera(force=True)

    def _update_camera(self, force: bool = False) -> None:
        visible = [
            geom for k, geom in self.geometries.items()
            if self.checkboxes.get(k, None) and self.checkboxes[k].checked
        ]
        if not visible:
            if force:
                self.scene_widget.scene.show_axes(True)
            return

        bbox = visible[0].get_axis_aligned_bounding_box()
        for geom in visible[1:]:
            bbox += geom.get_axis_aligned_bounding_box()
        self.scene_widget.scene.show_axes(True)
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def run(self) -> None:
        print("Visualization legend: target/reference = blue, perturbed source = red, GICP-aligned = green.")
        print("Close the viewer window to return to the terminal.")
        self.app.run()


def main() -> None:
    viewer = CloudViewer()
    viewer.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Viewer error: {exc}")
        sys.exit(1)
