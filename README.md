<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/logo.png" width="300">
</h1><br>

# MagTrack

## Overview
A fast python package for tracking symmetric beads in 
single-molecule magnetic tweezers experiments. Calculates the XY coordinates 
using a variety of common methods such as: center-of-mass and auto-convolution.
Calculates the Z coordinates with Z-LUT (lookup table).

* GPU accelerated (optional, requires a CUDA GPU)
* Sub-pixel XYZ coordinates
* Python notebook included with examples
* Documented and tested
* Only depends on Numpy, Scipy and Cupy

## Install
### Pre-requisites
This project was developed with the follow versions. It may work with older versions of Python and the required packages.
* Python >=3.11
* Numpy >=2.2
* Cupy-cuda12x >=13.4
* For GPU acceleration with CuPy you must have an CUDA complinat GPU and the CUDA toolkit installed. MagTrack can run without a GPU or toolkit but will only use the CPU.

### Instructions
1. ```pip install git+https://github.com/7jameslondon/MagTrack```
2. Setup Nvida GPU with Cupy https://docs.cupy.dev/en/stable/install.html

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