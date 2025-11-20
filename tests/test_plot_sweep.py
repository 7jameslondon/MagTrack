import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.accuracy import plot_sweep


class BeadSimulationSweep:
    """Small helper to fabricate sweep outputs for plotting tests."""

    @staticmethod
    def generate(output_dir: Path, n_images: int = 4, size: int = 8) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        combinations: list[dict[str, object]] = []
        images: dict[str, np.ndarray] = {}
        base = np.linspace(0.0, 1.0, num=size * size, dtype=np.float64).reshape(
            size, size, 1
        )

        for idx in range(n_images):
            key = f"image_{idx}"
            combinations.append(
                {
                    "image_path": f"images/{key}.npy",
                    "radius_nm": 1000.0 + idx,
                    "contrast_scale": 1.0 + 0.1 * idx,
                    "background_level": 0.5 + 0.05 * idx,
                }
            )
            images[key] = base + 0.01 * float(idx)

        np.savez(output_dir / "images.npz", **images)
        (output_dir / "metadata.json").write_text(
            json.dumps({"combinations": combinations}, indent=2)
        )
        return output_dir


def test_plot_sweep_saves_overview(tmp_path: Path) -> None:
    sweep_dir = BeadSimulationSweep.generate(tmp_path / "sweep")
    output = tmp_path / "overview.png"

    fig = plot_sweep.plot_sweep_directory(
        sweep_dir, max_images=3, output=output, show=False
    )

    assert output.exists()
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_load_sweep_resolves_image_keys(tmp_path: Path) -> None:
    sweep_dir = BeadSimulationSweep.generate(tmp_path / "sweep_keys", n_images=2)
    images = plot_sweep.load_sweep_images(sweep_dir)

    assert [image.key for image in images] == ["image_0", "image_1"]
    assert all(img.image.dtype == np.float64 for img in images)

    figure = plot_sweep.plot_sweep_images(images, output=None, show=False)
    assert len(figure.axes) == 2
    plt.close(figure)
