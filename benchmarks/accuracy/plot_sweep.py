"""TkAgg-based plotting utility for accuracy benchmark sweeps."""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
import tkinter as tk
from tkinter import ttk
from typing import Mapping

import matplotlib
import numpy as np
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from benchmarks.accuracy.sweep_loader import SweepData, SweepImage


_WRAP_WIDTH = 40


def _format_parameters(parameters: Mapping[str, object], fallback: str) -> str:
    if not parameters:
        return fallback
    joined = ", ".join(f"{key}={value}" for key, value in parameters.items())
    return textwrap.fill(joined, width=_WRAP_WIDTH)


def _compute_grid_shape(total: int) -> tuple[int, int]:
    if total <= 0:
        return 1, 1
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    return rows, cols


def _plot_images(ax_array: np.ndarray, images: list[SweepImage]) -> None:
    for index, sweep_image in enumerate(images):
        row, col = divmod(index, ax_array.shape[1])
        ax = ax_array[row, col]
        array = sweep_image.image.squeeze()
        ax.imshow(array, cmap="gray", vmin=0.0, vmax=1.0)
        title = _format_parameters(sweep_image.parameters, sweep_image.key)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for index in range(len(images), ax_array.size):
        row, col = divmod(index, ax_array.shape[1])
        ax_array[row, col].axis("off")


def _build_figure(sweep_data: SweepData) -> plt.Figure:
    total_images = len(sweep_data.images)
    rows, cols = _compute_grid_shape(total_images)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    ax_array = np.atleast_2d(axes).reshape(rows, cols)

    _plot_images(ax_array, sweep_data.images)
    fig.tight_layout()
    return fig


def _embed_figure_in_scrollable_canvas(root: tk.Tk, figure: plt.Figure) -> None:
    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(container)
    h_scrollbar = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=canvas.xview)
    v_scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)

    canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame_configure(event: tk.Event) -> None:  # type: ignore[type-arg]
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", _on_frame_configure)

    def _on_canvas_configure(event: tk.Event) -> None:  # type: ignore[type-arg]
        canvas.itemconfigure(canvas_window, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)

    figure_canvas = FigureCanvasTkAgg(figure, master=frame)
    widget = figure_canvas.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)


def _on_close(root: tk.Tk) -> None:
    root.quit()
    root.destroy()
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        default="default",
        help="Name of the sweep directory under benchmarks/accuracy/sweeps to load.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sweep_data = SweepData.load(args.sweep)

    figure = _build_figure(sweep_data)

    root = tk.Tk()
    root.title(f"Accuracy sweep: {sweep_data.sweep_name}")
    root.protocol("WM_DELETE_WINDOW", lambda: _on_close(root))
    _embed_figure_in_scrollable_canvas(root, figure)
    root.mainloop()


if __name__ == "__main__":
    main()
