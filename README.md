# MagTrack

## Overview
A simple yet fast python package for tracking of symmetric beads in 
single-molecule magnetic tweezers experiments. Calculates the XY coordinates 
using a variety of common methods such as: center-of-mass and auto-convolution.
Calculates the Z coordinates with Z-LUT (lookup table).

* Python notebook included with examples
* Easy to read and modify code
* Documented and tested
* Sub-pixel XYZ coordinates
* GPU acceleration (optional, requires a CUDA GPU)
* Only requires Numpy and Cupy

## Install
### Pre-requisites
* Python >=3.12 (untested on older versions)
* Numpy >=2.2.3
* Cupy-cuda12x >=13.4.0

```pip install magtrack @ git+https://github.com/7jameslondon/MagTrack```

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