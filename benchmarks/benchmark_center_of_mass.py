import itertools

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark as cupy_benchmark

import magtrack


def benchmark_center_of_mass():
    n_images = [128, 256, 512, 1024, 2048, 4096]
    widths = [64, 128, 256, 512]
    use_gpu = [False, True]

    mean_times = {}
    for n, w, g in list(itertools.product(n_images, widths, use_gpu)):
        # Create fake images
        stack = np.random.rand(w, w, n).astype(np.float32)
        if g:
            stack = cp.asarray(stack)

        # Repeatedly test speed
        n_warmup = 10 if g else 2
        results = cupy_benchmark(
            magtrack.center_of_mass,
            args=(stack, ),
            kwargs={'background': 'none'},
            max_duration=30,
            n_repeat=30,
            n_warmup=n_warmup,
        )
        if g:
            t = np.mean(results.gpu_times.squeeze() + results.cpu_times)
        else:
            t = np.mean(results.cpu_times)
        mean_times[(n, w, g)] = t

        # Update plot
        if not plt.fignum_exists(1):
            fig, axs = plt.subplots(len(widths), len(use_gpu), figsize=(8, 8))
        else:
            fig = plt.gcf()
            axs = np.array(fig.axes).reshape(len(widths), len(use_gpu))
        for key in mean_times:
            width_idx = widths.index(key[1])
            gpu_idx = use_gpu.index(key[2])
            label = 'GPU' if key[2] else 'CPU'
            color = 'g' if key[2] else 'b'
            axs[width_idx, gpu_idx].plot(
                key[0],
                mean_times[key],
                '.',
                markersize=10,
                clip_on=False,
                zorder=100,
                label=label,
                color=color
            )
        for i, w in enumerate(widths):
            for j, g in enumerate(use_gpu):
                g_srt = 'GPU' if g else 'CPU'
                axs[i, j].set_title(f'{g_srt} {w}x{w}')
                axs[i, j].autoscale()
                axs[i, j].autoscale_view()
                axs[i, j].set_ylim(0, None)
                axs[i, j].set_xlim(0, n_images[-1] * 1.05)
                if (i + 1) < len(widths):
                    axs[i, j].set_xticks([])
                axs[i, j].relim()
        #plt.legend()
        #plt.tight_layout()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)

    print('Done')
    plt.show()


if __name__ == "__main__":
    benchmark_center_of_mass()
