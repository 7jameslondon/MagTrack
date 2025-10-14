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
This project was developed with the follow versions. It may work with older versions of Python and the required packages but has not been tested.
* Python >=3.11
* Numpy >=2.2
* Cupy-cuda12x >=13.4

```pip install git+https://github.com/7jameslondon/MagTrack```

## Usage
```
import magtrack
import tifffile

stack = tifffile.imread('my_video.tiff')
x, y, z, profiles = magtrack.stack_to_xyzp(stack)
```
Download the notebook file to see more examples.

## Development

To format python files run: ```yapf main.py -i``` ```yapf .\magtrack\ -i -r```

To build the wheel: ```pip wheel . --no-deps```