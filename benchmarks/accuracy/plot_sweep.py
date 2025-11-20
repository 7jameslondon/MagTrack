"""Plot images from a sweep using matplotlib's Tk backend."""

from __future__ import annotations

import argparse
import math
import textwrap
import tkinter as tk
from typing import Any

import matplotlib

# Tk backend must be selected before importing pyplot-related modules.
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from benchmarks.accuracy.sweep_loader import SweepData, load_sweep


def _format_title(image_key: str, values: dict[str, Any]) -> str:
    if not values:
        return image_key
    parts = [f"{key}: {value}" for key, value in values.items()]
    return f"{image_key}\n" + "\n".join(textwrap.fill(part, width=50) for part in parts)


def _prepare_image(image: Any) -> Any:
    if getattr(image, "ndim", 0) == 3 and image.shape[-1] == 1:
        return image[:, :, 0]
    return image.squeeze()


def _build_figure(sweep: SweepData) -> Figure:
    image_keys = sweep.image_keys()
    if not image_keys:
        raise ValueError(f"No images found for sweep '{sweep.sweep_name}'.")

    total = len(image_keys)
    cols = min(3, total)
    rows = math.ceil(total / cols)
    fig = Figure(figsize=(cols * 4, rows * 4), layout="constrained")

    for idx, key in enumerate(image_keys):
        ax = fig.add_subplot(rows, cols, idx + 1)
        image = _prepare_image(sweep.images[key])
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        title = _format_title(key, sweep.combinations.get(key, {}))
        ax.set_title(title, wrap=True, fontsize=9)

    return fig


def _create_scrollable_window(fig: Figure, title: str) -> None:
    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root)
    scrollbar_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    frame = tk.Frame(canvas)
    frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    fig_canvas = FigureCanvasTkAgg(fig, master=frame)
    fig_canvas.draw()
    widget = fig_canvas.get_tk_widget()
    widget.pack(fill="both", expand=True)

    def _on_frame_configure(_event: tk.Event[tk.Misc]) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: tk.Event[tk.Misc]) -> None:
        canvas.itemconfigure(frame_id, width=event.width)

    frame.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    root.mainloop()


def plot_sweep(sweep_name: str) -> None:
    sweep = load_sweep(sweep_name)
    fig = _build_figure(sweep)
    _create_scrollable_window(fig, f"Sweep: {sweep.sweep_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        default="default",
        help="Name of the sweep in benchmarks/accuracy/sweeps to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_sweep(args.sweep)


if __name__ == "__main__":
    main()
