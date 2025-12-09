# pyoperon

[![License](https://img.shields.io/github/license/heal-research/pyoperon)](https://github.com/heal-research/pyoperon/blob/master/LICENSE)
[![Build-linux](https://github.com/heal-research/pyoperon/actions/workflows/build-linux.yml/badge.svg?branch=main)](https://github.com/heal-research/pyoperon/actions/workflows/build-linux.yml)
[![Build-macos](https://github.com/heal-research/pyoperon/actions/workflows/build-macos.yml/badge.svg?branch=main)](https://github.com/heal-research/pyoperon/actions/workflows/build-linux.yml)
[![Matrix chat](https://matrix.to/img/matrix-badge.svg)](https://matrix.to/#/#operon:matrix.org)

**pyoperon** is the python bindings library of [**Operon**](https://github.com/heal-research/operon), a modern C++ framework for symbolic regression developed by [Heal-Research](https://github.com/heal-research) at the University of Applied Sciences Upper Austria.

A scikit-learn regressor is also available:
```python
from pyoperon.sklearn import SymbolicRegressor
```

The [example](https://github.com/heal-research/pyoperon/tree/main/example) folder contains sample code for using either the Python bindings directly or the **pyoperon.sklearn** module.

# Installation

New releases are published on [github](https://github.com/heal-research/pyoperon/releases/) and on [PyPI](https://pypi.org/project/pyoperon/).

Most of the time `pip install pyoperon` should be enough.

## Building from source

### Conda/Mamba

1. Clone the repository
```
git clone https://github.com/heal-research/pyoperon.git
cd pyoperon
```

2. Install and activate the environment (replace micromamba with your actual program)
```
micromamba env create -f environment.yml
micromamba activate pyoperon
```

3. Install the dependencies
```
export CC=${CONDA_PREFIX}/bin/clang
export CXX=${CONDA_PREFIX}/bin/clang++
python script/dependencies.py
```

4. Install `pyoperon`
```
pip install .
```

### Nix

Use this [environment](https://github.com/foolnotion/poetryenv) created with [poetry2nix](https://github.com/nix-community/poetry2nix)

```
nix develop github:foolnotion/poetryenv --no-write-lock-file
```

This will install operon and dependencies. Modify the flake file if you need additional python libraries (see https://github.com/nix-community/poetry2nix#how-to-guides)


# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.
