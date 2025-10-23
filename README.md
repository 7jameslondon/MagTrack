<h1 align="center">
<img src="https://raw.githubusercontent.com/7jameslondon/MagTrack/refs/heads/master/logo.png" width="300">
</h1><br>

MagTrack is a Python package for tracking symmetric beads in 
single-molecule magnetic tweezers experiments. 

* GPU accelerated (optional, requires a CUDA GPU)
* Sub-pixel XYZ coordinates
* Python notebook included with examples
* Documented, tested and benchmarked
* Only depends on Numpy, Scipy and Cupy
* Activly maintained

Try a demo in a Google Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## Install
### Pre-requisites
MagTrack has been tested with the following requirments; It may work with older versions.
* Python >=3.11
* Numpy >=2.2
* Cupy-cuda12x >=13.4
* MagTrack can run on a CPU or GPU. But to run on a GPU it must be CUDA compliant GPU with the CUDA toolkit installed. This is free and easy to install for most NVIDA GPUs.

### Instructions
```pip install magtrack```

Optional: For GPU acceleration on a computer with an NVIDA CUDA GPU you may need install the CUDA Toolkit for Cupy. See details at https://docs.cupy.dev/en/stable/install.html

## Usage
```
import magtrack
import tifffile # use whatever library you prefer to prepare/import your video into Python as an 3D-array

# Import a video
stack = tifffile.imread('my_video.tiff')

# Run the full defualt XYZ pipeline
x, y, z, profiles = magtrack.stack_to_xyzp(stack)

# Or mix and match algorithms your prefer
x, y = center_of_mass(stack)
x, y = auto_conv(stack)
profiles = magtrack.fft_profile(stack)

```
### More Examples
You can see more example of how to use MagTrack in [this notebook](https://github.com/7jameslondon/MagTrack/blob/master/examples/examples.ipynb).
You can download it and run it on your computer.
Or try it out with Google Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/7jameslondon/MagTrack/blob/master/examples/examples.ipynb)

## Documentation
### User Guide
Coming soon
### Documentation
Coming soon
