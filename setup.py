import os, sys, subprocess
from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="pyoperon",
    version="0.3.6",
    description="python bindings for the operon library",
    author='Bogdan Burlacu',
    license="MIT",
    packages=['pyoperon'],
    python_requires=">=3.7",
)
