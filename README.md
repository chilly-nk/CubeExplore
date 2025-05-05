# CubeExplore

**CubeExplore** is a lightweight Python module for exploring 3D and **4D hyperspectral image (HSI)** data.  
It provides utilities for loading, processing, visualizing, and interacting with spectral cubes.

## Features
- Load HSI cubes in formats:
    - .im3
    - .bin / .hdr
    - .tif (recommended usage)
- Explore pixel spectra and excitation-emission matrices (EEMs)
- Quick ROI (Region of Interest) tools
- etc. (to be described)

## Installation

### 1. Clone the repository

```bash
git clone --branch v1.0 https://github.com/chilly-nk/CubeExplore.git
```

### 2. System requirements

```bash
apt update
apt install -y maven
```
If you're using **Google Colab**:
- prefix the commands with an explanation mark: `!apt install -y maven`
- or use a shell cell to run all commands at once as shell commands:
```python
%%shell
apt update
apt install -y maven
```

### 3. Install the module

To install the module **along with its Python dependencies**, navigate to the project folder and run:

```bash
pip install .
```
If you're using **Google Colab**, prefix the command with an exclamation mark: `!pip install .`

## Usage

## License