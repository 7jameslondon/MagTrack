<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/assets/logo.png" width="300">
</h1><br>

[![PyPi](https://img.shields.io/pypi/v/magtrack)](https://pypi.org/project/magtrack/)
[![Docs](https://img.shields.io/readthedocs/magtrack/latest.svg)](https://magtrack.readthedocs.io)
[![Testing](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml/badge.svg)](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml)
[![Paper](https://img.shields.io/badge/DOI-10.1101/2025.10.31.685671-blue)](https://doi.org/10.1101/2025.10.31.685671)

[MagTrack](https://github.com/7jameslondon/MagTrack/) is a Python library for tracking symmetric beads in [single-molecule magnetic tweezers](https://doi.org/10.1007/978-1-0716-3377-9_18) experiments. 

* Sub-pixel XYZ coordinates
* GPU accelerated (optional, requires a CUDA GPU)
* Python notebook included with [examples](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)
* [Documented](https://magtrack.readthedocs.io/en/stable/), [tested](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml), and [benchmarked](https://magtrack.readthedocs.io/en/stable/benchmarking.html)
* Only depends on [NumPy](https://numpy.org), [SciPy](https://scipy.org), and [CuPy](https://cupy.dev)

Try a demo in a Google Colab notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

<h3 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/assets/demo.gif" width="600">
</h3>

## ⏳ Install
* Windows, MacOS, or Linux (note: MacOS does not support CUDA GPU acceleration)
* [Python](https://www.python.org/downloads/) >=3.9

### Method 1: CPU Only installation (easier)
This method works for MacOS, Windows, or Linux but will only use your CPU.
You can use pip or conda to install MagTrack.
```
pip install magtrack[cpu]
```
conda *Coming soon*


### Method 2: GPU or CPU installation (a little harder)
This method work for Windows or Linux (not MacOS).
MagTrack can run on a CPU or GPU. 
Your computer must have a CUDA GPU (most NVIDIA GPUs will work).

1. Install the free [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). You can pick versions 11, 12, or 13.
   * You can check what version you have installed by running "nvcc -V" in a terminal (Command Prompt).
2. Install MagTrack with pip or conda.
   * Pip: Use the pip command corresponding to your CUDA Toolkit version.

       | CUDA Toolkit Version      | Pip Command                      |
       |---------------------------|----------------------------------|
       | CUDA Toolkit Version 11.x | ```pip install magtrack[cu11]``` |
       | CUDA Toolkit Version 12.x | ```pip install magtrack[cu12]``` |
       | CUDA Toolkit Version 13.x | ```pip install magtrack[cu13]``` |

   * Conda: *Coming soon*




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
Try it out with Google Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)
Or you can [download](https://github.com/7jameslondon/MagTrack/blob/master/examples/examples.ipynb) it and run it on your computer with [Jupyter Notebook](https://jupyter.org/install).

## 📖 Documentation
View the full guide to MagTrack at [magtrack.readthedocs.io](https://magtrack.readthedocs.io)

## 💬 Support
Report issues and make requests on the [GitHub issue tracker](https://github.com/7jameslondon/MagTrack/issues).<br><br>
Having trouble? Need help? Have suggestions? Want to contribute?<br>
Email us at magtrackandmagscope@gmail.com

