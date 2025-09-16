"""Interactive viewer for Argoverse `city_SE3_egovehicle.feather` pose files."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

MAX_ROWS_DISPLAY = 500


@dataclass
class PoseSummary:
    path: Path
    rows: int
    start_ts: int
    end_ts: int
    duration_s: float
    mean_dt_ms: float
    translation_min: np.ndarray
    translation_max: np.ndarray

    @classmethod
    def from_frame(cls, path: Path, frame: pd.DataFrame) -> "PoseSummary":
        timestamps = frame["timestamp_ns"].to_numpy(np.int64)
        tx = frame[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64)
        if timestamps.size == 0:
            raise ValueError("Pose frame is empty.")
        start_ts = int(timestamps.min())
        end_ts = int(timestamps.max())
        duration_s = (end_ts - start_ts) / 1e9
        if timestamps.size > 1:
            steps = np.diff(np.sort(timestamps))
            mean_dt_ms = float(np.mean(steps) / 1e6)
        else:
            mean_dt_ms = 0.0
        return cls(
            path=path,
            rows=int(frame.shape[0]),
            start_ts=start_ts,
            end_ts=end_ts,
            duration_s=duration_s,
            mean_dt_ms=mean_dt_ms,
            translation_min=tx.min(axis=0),
            translation_max=tx.max(axis=0),
        )

    def format_summary(self) -> str:
        start = dt.datetime.fromtimestamp(self.start_ts / 1e9)
        end = dt.datetime.fromtimestamp(self.end_ts / 1e9)
        return (
            f"Rows: {self.rows}\n"
            f"Start: {start.isoformat()}\n"
            f"End:   {end.isoformat()}\n"
            f"Duration: {self.duration_s:.2f}s\n"
            f"Mean Δt: {self.mean_dt_ms:.2f} ms\n"
            f"Translation X: [{self.translation_min[0]:.3f}, {self.translation_max[0]:.3f}] m\n"
            f"Translation Y: [{self.translation_min[1]:.3f}, {self.translation_max[1]:.3f}] m\n"
            f"Translation Z: [{self.translation_min[2]:.3f}, {self.translation_max[2]:.3f}] m"
        )


class CityPoseViewer(tk.Tk):
    def __init__(self, initial_path: Optional[Path] = None) -> None:
        super().__init__()
        self.title("Argoverse Pose Viewer")
        self.geometry("1100x700")
        self._path_var = tk.StringVar()

        self._build_widgets()

        if initial_path:
            self.load_file(initial_path)

    def _build_widgets(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        open_btn = ttk.Button(top_frame, text="Open .feather", command=self.browse_file)
        open_btn.pack(side=tk.LEFT)

        path_entry = ttk.Entry(top_frame, textvariable=self._path_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        summary_frame = ttk.LabelFrame(self, text="Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._summary_label = ttk.Label(summary_frame, justify=tk.LEFT, anchor=tk.W, text="No file loaded.")
        self._summary_label.pack(fill=tk.X, padx=10, pady=10)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m")
        self._tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        for col in columns:
            self._tree.heading(col, text=col)
            self._tree.column(col, anchor=tk.CENTER, width=120)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    def browse_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select city_SE3_egovehicle.feather",
            filetypes=(("Feather", "*.feather"), ("All files", "*.*")),
        )
        if not filename:
            return
        self.load_file(Path(filename))

    def load_file(self, path: Path) -> None:
        try:
            table = feather.read_table(path)
            frame = table.to_pandas()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read {path}:\n{exc}")
            return

        try:
            summary = PoseSummary.from_frame(path, frame)
            summary_text = summary.format_summary()
        except Exception as exc:
            summary_text = f"Unable to compute summary: {exc}"

        self._path_var.set(str(path))
        self._summary_label.configure(text=summary_text)
        self._populate_tree(frame)

    def _populate_tree(self, frame: pd.DataFrame) -> None:
        self._tree.delete(*self._tree.get_children())
        columns = list(frame.columns)
        self._tree.configure(columns=columns)
        for col in columns:
            self._tree.heading(col, text=col)
            width = 140 if col.endswith("_m") else 180 if col == "timestamp_ns" else 120
            self._tree.column(col, width=width, anchor=tk.CENTER)

        subset = frame.head(MAX_ROWS_DISPLAY)
        for _, row in subset.iterrows():
            values = [self._format_value(v) for v in row]
            self._tree.insert("", tk.END, values=values)

        if len(frame) > MAX_ROWS_DISPLAY:
            messagebox.showinfo(
                "Notice",
                f"Displaying first {MAX_ROWS_DISPLAY} rows of {len(frame)}. Export or filter in pandas for the full set.",
            )

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)


def launch(initial_path: Optional[Path]) -> None:
    app = CityPoseViewer(initial_path=initial_path)
    app.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI to explore city_SE3_egovehicle.feather pose files.")
    parser.add_argument("--file", type=Path, help="Optional path to open on launch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial = args.file if args.file and args.file.exists() else None
    if args.file and initial is None:
        print(f"Warning: file not found: {args.file}", file=sys.stderr)
    launch(initial)


if __name__ == "__main__":
    main()
