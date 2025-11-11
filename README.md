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
* GPU accelerated (optional, requires a NVIDIA CUDA GPU)
* Python notebook included with [examples](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)
* [Documented](https://magtrack.readthedocs.io/en/stable/), [tested](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml), and [benchmarked](https://magtrack.readthedocs.io/en/stable/benchmarking.html)
* Only depends on [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [CuPy](https://cupy.dev)

Try a demo in a Google Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

<h3 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/assets/demo.gif" width="600">
</h3>

## ⏳ Install
### Pre-requisites
* Operating System: Windows or Linux or MacOS (MacOS does not support NVIDIA GPU acceleration)
* [Python](https://www.python.org/downloads/) >=3.9
* [NumPy](https://numpy.org) >=1.26
* [SciPy](https://scipy.org) >=1.11.1
* (Optional, GPU acceleration) [CuPy-CUDA12x](https://cupy.dev) >=13.0
* (Optional, GPU acceleration) [CUDA Toolkit](https://docs.cupy.dev/en/stable/install.html). This is free and easy to install for computers with most NVIDIA GPUs.

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

