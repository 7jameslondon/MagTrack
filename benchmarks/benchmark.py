import itertools

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark as cupy_benchmark

import magtrack

def benchmark_center_of_mass():
    def gen_center_of_mass(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.float64)
        if g:
            stack = cp.asarray(stack)
        return (stack,)

    run_benchmark(magtrack.center_of_mass,
                  gen_center_of_mass,
                  kwargs={'background': 'none'})

def benchmark_auto_conv():
    def gen_auto_conv(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.float64)
        x = np.random.rand(n).astype(np.float64)
        y = np.random.rand(n).astype(np.float64)
        if g:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)
        return (stack, x, y)

    run_benchmark(magtrack.auto_conv,
                  gen_auto_conv)

def benchmark_auto_conv_multiline_para_fit():
    def gen_auto_conv_multiline_para_fit(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.float64)
        x = np.random.rand(n).astype(np.float64)
        y = np.random.rand(n).astype(np.float64)
        if g:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)
        return (stack, x, y)

    run_benchmark(magtrack.auto_conv_multiline_para_fit,
                  gen_auto_conv_multiline_para_fit)

def benchmark_fft_profile():
    def gen_fft_profile(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.float64)
        x = np.random.rand(n).astype(np.float64)
        y = np.random.rand(n).astype(np.float64)
        if g:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)
        return (stack, x, y)

    run_benchmark(magtrack.fft_profile,
                  gen_fft_profile)

def benchmark_radial_profile():
    def gen_radial_profile(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.float64)
        x = np.random.rand(n).astype(np.float64)
        y = np.random.rand(n).astype(np.float64)
        if g:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)
        return (stack,x,y)

    run_benchmark(magtrack.radial_profile,
                  gen_radial_profile)

def benchmark_stack_to_xyzp():
    def gen_stack_to_xyzp(n, w, g):
        stack = np.random.rand(w, w, n).astype(np.uint8)
        return (stack,)

    run_benchmark(magtrack.stack_to_xyzp,
                  gen_stack_to_xyzp)

def run_benchmark(func, arg_generator,
                  n_images=[128, 256, 512, 1024, 2048],
                  widths=[128, 256, 512],
                  use_gpu=[True],
                  kwargs={}):

    mean_times = {}
    for n, w, g in list(itertools.product(n_images, widths, use_gpu)):
        # Create fake data
        args = arg_generator(n, w, g)

        # Repeatedly test speed
        n_warmup = 10 if g else 2
        results = cupy_benchmark(
            func,
            args=args,
            kwargs={},
            max_duration=1,
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
            axs = np.array(fig.axes)
        axs = axs.reshape(len(widths), len(use_gpu))
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
                clip_on=True,
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
                axs[i, j].set_ylim(0, 1)
                axs[i, j].set_xlim(0, n_images[-1] * 1.05)
                if (i + 1) < len(widths):
                    axs[i, j].set_xticks([])
                axs[i, j].relim()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)

    print('Done')
    plt.show()


if __name__ == "__main__":
    benchmark_fft_profile()
