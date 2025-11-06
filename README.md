<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/assets/logo.png" width="300">
</h1><br>

[![PyPi](https://img.shields.io/pypi/v/magtrack.svg)](https://pypi.org/project/magtrack/)
[![Downloads](https://img.shields.io/pypi/dm/MagTrack)](https://pypi.org/project/magtrack/)
[![Docs](https://img.shields.io/readthedocs/magtrack/latest.svg)](https://magtrack.readthedocs.io)
[![Paper](https://img.shields.io/badge/DOI-10.1101/2025.10.31.685671-blue)](https://doi.org/10.1101/2025.10.31.685671)
[![Python package](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml/badge.svg)](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml)

MagTrack is a Python package for tracking symmetric beads in single-molecule magnetic tweezers experiments. 

* Sub-pixel XYZ coordinates
* GPU accelerated (optional, requires a CUDA GPU)
* Python notebook included with [examples](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)
* [Documented](https://magtrack.readthedocs.io/en/stable/), [tested](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml), and [benchmarked](https://github.com/7jameslondon/MagTrack/tree/master/benchmarks)
* Only depends on [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [CuPy](https://cupy.dev/)

Try a demo in a Google Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

<h3 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/assets/demo.gif" width="600">
</h3>

## ⏳ Install
### Pre-requisites
* Python >=3.9
* NumPy >=1.26
* SciPy >=1.11.1
* (Optional, but needed for GPU acceleration) CuPy-CUDA12x >=13.0
* Windows or Linux or MacOS (MacOS does not support NVIDIA GPU acceleration)
* MagTrack can run on a CPU or GPU. But GPU execution requires a CUDA-compliant GPU with the CUDA Toolkit installed. This is free and easy to install for most NVIDIA GPUs.

### Instructions
```
pip install magtrack[gpu]
```
Or without CuPy
```
pip install magtrack
```

Optional: For GPU acceleration on a computer with an NVIDIA CUDA GPU, you may need to install the CUDA Toolkit for CuPy. See details at https://docs.cupy.dev/en/stable/install.html

## ⚒ Usage
```
import magtrack

# Run the full default XYZ pipeline
x, y, z, profiles = magtrack.stack_to_xyzp(stack)

# Or make your own pipeline from algorithms you prefer
x, y = magtrack.center_of_mass(stack)
x, y = magtrack.auto_conv_sub_pixel(stack, x, y)
profiles = magtrack.fft_profile(stack)
...
```
### More Examples
You can see more examples of how to use MagTrack in [this notebook](https://github.com/7jameslondon/MagTrack/blob/master/examples/examples.ipynb).
You can download it and run it on your computer.
Or try it out with Google Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## 📖 Documentation
View the full documentation at [magtrack.readthedocs.io](https://magtrack.readthedocs.io)

## 💬 Support
Having trouble? Need help? Have suggestions? Or want to contribute code?<br>
Report issues and make requests on the [GitHub issue tracker](https://github.com/7jameslondon/MagTrack/issues)<br>
Or email us at magtrackandmagscope@gmail.com

## 📈 Benchmarks and performance tracking

Run the automated benchmark orchestrator to execute every available benchmark suite, capture
system metadata, and generate a comparison plot against previous runs:

```bash
python -m benchmarks.run_all
```

Each invocation records CPU and GPU runtimes (when CuPy is available) alongside host metadata
(hostname, OS, Python and dependency versions, and optional `psutil`/CuPy hardware details).
The results are written to JSON logs inside `benchmarks/logs/<system-id>/<timestamp>/results.json`,
where `<system-id>` encodes the hostname, platform, and Python version, and `<timestamp>` uses a
UTC timestamp such as `20240521T154500Z`.

After every run the orchestrator also refreshes a consolidated CSV at
`benchmarks/logs/combined_results.csv` that aggregates every historical benchmark entry. When a
GUI backend is available a Matplotlib bar chart is displayed comparing the most recent run against
the historical mean for each benchmark/backend pair. The raw log files can be version-controlled to
track long-term performance regressions or improvements.
