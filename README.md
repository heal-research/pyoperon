# pyoperon

[![License](https://img.shields.io/github/license/heal-research/pyoperon?style=flat)](https://github.com/heal-research/pyoperon/blob/master/LICENSE)
[![build-linux](https://github.com/heal-research/pyoperon/actions/workflows/build-linux.yml/badge.svg)](https://github.com/heal-research/pyoperon/actions/workflows/build-linux.yml)
[![Gitter chat](https://badges.gitter.im/operongp/gitter.png)](https://gitter.im/operongp/community)

**pyoperon** is the python bindings library of [**Operon**](https://github.com/heal-research/operon), a modern C++ framework for symbolic regression developed by [Heal-Research](https://github.com/heal-research) at the University of Applied Sciences Upper Austria.

A scikit-learn regressor is also available:
```python
from pyoperon.sklearn import SymbolicRegressor
```

The [examples](https://github.com/heal-research/pyoperon/examples) folder contains sample code for using either the Python bindings directly or the **pyoperon.sklearn** module.

# Installation

The easiest way to install **pyoperon** is with **pip**:
```
pip install pyoperon
```
Note that the pyoperon python module links against the shared python interpreter library (libpython.so), so it's important that this library is in the path (e.g., `LD_LIBRARY_PATH` on linux).

Another way to get **pyoperon** is via the [nix package manager](https://nixos.org/). Nix can be installed on other Linux distributions in a few easy steps:

1. [Install nix](https://nixos.org/manual/nix/stable/installation/installing-binary.html) and enable flake support in `~/.config/nix/nix.conf`:
   ```
   experimental-features = nix-command flakes
   ```
2. Install **pyoperon**:
   ```
   nix develop github:heal-research/pyoperon --no-write-lock-file
   ```

Upon completion of the last command, the `$PYTHONPATH` will be updated and **pyoperon** will pe available for use. Note that as opposed to PyPI releases, the nix flake will always build the latest development version from github.

Alternatively, one can also clone https://github.com/heal-research/pyoperon.git and run `nix develop` from within the cloned path.

# Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) document.
