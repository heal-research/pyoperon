# pyoperon

[![License](https://img.shields.io/github/license/heal-research/pyoperon)](https://github.com/heal-research/pyoperon/blob/master/LICENSE)
[![Build-linux](https://github.com/heal-research/pyoperon/actions/workflows/build-wheels.yml/badge.svg?branch=main)](https://github.com/heal-research/pyoperon/actions/workflows/build-wheels.yml)
[![Build-macos](https://github.com/heal-research/pyoperon/actions/workflows/build-wheels.yml/badge.svg?branch=main)](https://github.com/heal-research/pyoperon/actions/workflows/build-wheels.yml)
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

```bash
git clone https://github.com/heal-research/pyoperon.git
cd pyoperon
```

2. Install and activate the environment (replace `micromamba` with your package manager)

```bash
micromamba env create -f environment.yml
micromamba activate pyoperon
```

3. Build the C++ dependencies and install `pyoperon`

```bash
export CC=clang
export CXX=clang++
python script/dependencies.py
pip install . --no-build-isolation
```

### Nix

The repository includes a `flake.nix` with a development shell that provides all C++ and Python dependencies.

1. Clone the repository

```bash
git clone https://github.com/heal-research/pyoperon.git
cd pyoperon
```

2. Enter the dev shell and install `pyoperon` into a virtual environment

```bash
nix develop .#pyenv
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install scikit-build-core
pip install --no-build-isolation .
```

3. Run the tests (optional)

```bash
pip install --no-build-isolation '.[test]'
pytest tests/ -v
```


# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.
