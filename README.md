<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/logo.png" width="300">
</h1><br>

MagTrack is an optimized python package for tracking symmetric beads in 
single-molecule magnetic tweezers experiments. 

* GPU accelerated (optional, requires a CUDA GPU)
* Sub-pixel XYZ coordinates
* Python notebook included with examples
* Documented and tested
* Only depends on Numpy, Scipy and Cupy

Try a demo in a Google Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## Install
### Pre-requisites
This project was developed with the following versions. It may work with older versions of Python and the required packages.
* Python >=3.11
* Numpy >=2.2
* Cupy-cuda12x >=13.4
* For GPU acceleration with CuPy you must have an CUDA complinat GPU and the CUDA toolkit installed. MagTrack can run without a GPU or toolkit but will only use the CPU.

### Instructions
1. ```pip install magtrack```
2. Setup CUDA GPU with Cupy https://docs.cupy.dev/en/stable/install.html

## Usage
```
import magtrack
import tifffile

stack = tifffile.imread('my_video.tiff')
x, y, z, profiles = magtrack.stack_to_xyzp(stack)
```
Download the notebook file to see more examples.

## Development

To build the wheel: ```pip wheel . --no-deps```

To publish on PyPI
https://packaging.python.org/en/latest/tutorials/packaging-projects/
In a development environment (no particular packages are required), run the following:
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```