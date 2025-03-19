import itertools
import os
import time
import tempfile

import cupy as cp
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark as cupy_benchmark

import magtrack


def benchmark_save_stack_to_disk():
    STACK_SHAPE = (512, 512, 1000)
    STACK_TYPE = np.uint8
    N_REPEAT_TEST = 100

    # Create fake images
    stack = np.random.rand(*STACK_SHAPE).astype(STACK_TYPE)

    # Create temporary filepath
    temp_dir = tempfile.gettempdir()
    temp_filename = 'test_image.tif'
    temp_filepath = os.path.join(temp_dir, temp_filename)

    # Repeatedly test speed
    test_times = np.empty(N_REPEAT_TEST)
    for i in range(test_times.size):
        # Test speed
        start_time = time.perf_counter()
        tf.imwrite(temp_filepath, stack, imagej=True, metadata={'axes': 'TYX'})
        test_times[i] = time.perf_counter() - start_time

        # Delete file
        os.remove(temp_filepath)

    # Results
    print(f'Mean (seconds): {np.mean(test_times)}')
    print(f'Min (seconds): {np.min(test_times)}')
    print(f'Max (seconds): {np.max(test_times)}')
    plt.hist(test_times)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.show()


def benchmark_auto_conv():
    N_IMAGES = np.logspace(7, 13.3, num=10, base=2).astype(int)
    IMAGE_WIDTH = 256
    MAX_DURATION = 30
    N_WARMUP = 10
    USE_GPU = True

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images
        stack = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH,
                               n_images).astype(imageprocessing.FLOAT)
        x = np.zeros(n_images, dtype=imageprocessing.FLOAT) + (IMAGE_WIDTH / 2)
        y = x.copy()
        if USE_GPU:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)

        # Repeatedly test speed
        results = cupy_benchmark(
            imageprocessing.auto_conv,
            args=(stack, x, y),
            max_duration=MAX_DURATION,
            n_repeat=10,
            n_warmup=N_WARMUP,
        )
        # Results
        if USE_GPU:
            mean_times.append(
                np.mean(results.gpu_times.squeeze() + results.cpu_times)
            )
        else:
            mean_times.append(np.mean(results.cpu_times))

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            '.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, 0.16)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


def benchmark_auto_conv_para_fit():
    N_IMAGES = np.logspace(7, 13.3, num=10, base=2).astype(int)
    IMAGE_WIDTH = 256
    MAX_DURATION = 30
    N_WARMUP = 10
    USE_GPU = True

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images
        stack = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH,
                               n_images).astype(imageprocessing.FLOAT)
        x = np.zeros(n_images, dtype=imageprocessing.FLOAT) + (IMAGE_WIDTH / 2)
        y = x.copy()
        if USE_GPU:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)

        # Repeatedly test speed
        results = cupy_benchmark(
            imageprocessing.auto_conv_para_fit,
            args=(stack, x, y),
            max_duration=MAX_DURATION,
            n_repeat=10,
            n_warmup=N_WARMUP,
        )
        # Results
        if USE_GPU:
            mean_times.append(
                np.mean(results.gpu_times.squeeze() + results.cpu_times)
            )
        else:
            mean_times.append(np.mean(results.cpu_times))

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            '.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, 0.21)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


def benchmark_radial_profile():
    N_IMAGES = np.logspace(7, 13.3, num=10, base=2).astype(int)
    IMAGE_WIDTH = 512
    MAX_DURATION = 30
    N_REPEAT = 100
    N_WARMUP = 10
    USE_GPU = True

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images
        stack = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH,
                               n_images).astype(imageprocessing.FLOAT)
        x = np.zeros(n_images, dtype=imageprocessing.FLOAT) + (IMAGE_WIDTH / 2)
        y = x.copy()
        if USE_GPU:
            stack = cp.asarray(stack)
            x = cp.asarray(x)
            y = cp.asarray(y)

        # Repeatedly test speed
        results = cupy_benchmark(
            imageprocessing.radial_profile,
            args=(stack, x, y),
            max_duration=MAX_DURATION,
            n_repeat=N_REPEAT,
            n_warmup=N_WARMUP,
        )
        # Results
        if USE_GPU:
            mean_times.append(
                np.mean(results.gpu_times.squeeze() + results.cpu_times)
            )
        else:
            mean_times.append(np.mean(results.cpu_times))

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            'r.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, 4)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


def benchmark_stack_to_xyzp():
    N_IMAGES = np.linspace(100, 1000, num=10).astype(
        int
    )  #np.logspace(7, 12, num=12-7+1, base=2).astype(int)
    IMAGE_WIDTH = 512
    MAX_DURATION = 30
    N_REPEAT = 100
    N_WARMUP = 10
    USE_GPU = True

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images
        stack = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH,
                               n_images).astype(imageprocessing.FLOAT)
        zlut = np.random.rand((IMAGE_WIDTH // 4) + 1,
                              1000).astype(imageprocessing.FLOAT)
        if USE_GPU:
            stack = cp.asarray(stack)
            zlut = cp.asarray(zlut)

        # Repeatedly test speed
        results = cupy_benchmark(
            imageprocessing.stack_to_xyzp,
            args=(stack, zlut),
            max_duration=MAX_DURATION,
            n_repeat=N_REPEAT,
            n_warmup=N_WARMUP,
        )
        # Results
        if USE_GPU:
            mean_times.append(
                np.mean(results.gpu_times.squeeze() + results.cpu_times)
            )
        else:
            mean_times.append(np.mean(results.cpu_times))

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            'r.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, None)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


def benchmark_asarray():
    N_IMAGES = np.logspace(7, 13.3, num=10, base=2).astype(int)
    IMAGE_WIDTH = 512
    MAX_DURATION = 30
    N_WARMUP = 10

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images
        stack = np.random.rand(IMAGE_WIDTH, IMAGE_WIDTH,
                               n_images).astype(imageprocessing.FLOAT)

        # Repeatedly test speed
        results = cupy_benchmark(
            cp.asarray,
            args=(stack, ),
            max_duration=MAX_DURATION,
            n_repeat=10,
            n_warmup=N_WARMUP,
        )
        # Results
        mean_times.append(
            np.mean(results.gpu_times.squeeze() + results.cpu_times)
        )

        bps = stack.nbytes / mean_times[-1]
        print(f'Bytes/second {bps}')

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            '.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, None)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


def benchmark_get_array_module():

    stack = cp.random.rand(10).astype(imageprocessing.FLOAT)

    results = cupy_benchmark(
        cp.get_array_module,
        args=(stack, ),
        max_duration=30,
        n_repeat=100,
        n_warmup=10,
    )

    print(np.mean(results.gpu_times.squeeze() + results.cpu_times))


def bechmark_crop():
    N_IMAGES = np.logspace(1, 3, num=10, base=10).astype(int)
    IMAGE_WIDTH = 5120
    N_ROIS = 10
    ROI_START = 512
    ROI_END = 512
    MAX_DURATION = 30
    N_WARMUP = 1
    USE_GPU = False

    mean_times = []
    for n_images in N_IMAGES:
        # Create fake images and rois
        stack = np.empty((IMAGE_WIDTH, IMAGE_WIDTH, n_images), dtype=np.uint8)
        rois = np.ones((N_ROIS, 4), dtype=np.integer)
        rois[:, 0:4:2] = ROI_START
        rois[:, 1:4:2] = ROI_END
        if USE_GPU:
            stack = cp.asarray(stack)
            rois = cp.asarray(rois)

        # Repeatedly test speed
        results = cupy_benchmark(
            imageprocessing.crop_stack_to_rois,
            args=(stack, rois),
            max_duration=MAX_DURATION,
            n_repeat=10,
            n_warmup=N_WARMUP,
        )
        # Results
        if USE_GPU:
            mean_times.append(
                np.mean(results.gpu_times.squeeze() + results.cpu_times)
            )
        else:
            mean_times.append(np.mean(results.cpu_times))

        plt.clf()
        plt.plot(
            N_IMAGES[0:len(mean_times)],
            mean_times,
            '.',
            markersize=10,
            clip_on=False,
            zorder=100
        )
        plt.ylim(0, None)
        plt.xlim(0, N_IMAGES[-1] * 1.05)
        plt.draw()
        plt.show(block=False)
        plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    benchmark_center_of_mass()
