<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/logo.png" width="300">
</h1><br>

[![PyPi](https://img.shields.io/pypi/v/magtrack.svg)](https://pypi.org/project/magtrack/)
[![Docs](https://img.shields.io/readthedocs/magtrack/latest.svg)](https://magtrack.readthedocs.io/en/latest/)
[![Paper](https://img.shields.io/badge/DOI-INSERTDOI-blue)](
https://doi.org/DOI-INSERTDOI)
[![Python package](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml/badge.svg)](https://github.com/7jameslondon/MagTrack/actions/workflows/python-package.yml)

MagTrack is a Python package for tracking symmetric beads in single-molecule magnetic tweezers experiments. 

* GPU accelerated (optional, requires a CUDA GPU)
* Sub-pixel XYZ coordinates
* Python notebook included with examples
* Documented, tested, and benchmarked
* Only depends on NumPy, SciPy and CuPy
* Actively maintained

Try a demo in a Google Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## Install
### Pre-requisites
MagTrack has been tested with the following requirements (but it may work with older versions):
* Python >=3.9
* NumPy >=1.26
* SciPy >=1.11.1
* CuPy-CUDA12x >=13.0
* MagTrack can run on a CPU or GPU. But GPU execution requires a CUDA-compliant GPU with the CUDA Toolkit installed. This is free and easy to install for most NVIDIA GPUs.

### Instructions
```
pip install magtrack
```

Optional: For GPU acceleration on a computer with an NVIDIA CUDA GPU, you may need to install the CUDA Toolkit for CuPy. See details at https://docs.cupy.dev/en/stable/install.html

## Usage
```
import magtrack
import tifffile # use your preferred library to import a video

# Import a video
stack = tifffile.imread('my_video.tiff')

# Run the full default XYZ pipeline
x, y, z, profiles = magtrack.stack_to_xyzp(stack)

# Or make your own pipeline from algorithms you prefer
x, y = magtrack.center_of_mass(stack)
x, y = magtrack.auto_conv(stack, x, y)
profiles = magtrack.fft_profile(stack)

```
### More Examples
You can see more examples of how to use MagTrack in [this notebook](https://github.com/7jameslondon/MagTrack/blob/master/examples/examples.ipynb).
You can download it and run it on your computer.
Or try it out with Google Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## Documentation
View the full documentation at [magtrack.readthedocs.io](https://magtrack.readthedocs.io/en/latest/)

## Support
Report issues and ask questions on the [GitHub issue tracker](https://github.com/7jameslondon/MagTrack/issues)