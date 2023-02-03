import os, sys, subprocess
from skbuild import setup  # This line replaces 'from setuptools import setup'

if 'sdist' in sys.argv:
    cwd = os.getcwd()
    result = subprocess.call('./script/dependencies.sh', shell=True)
    if result != 0:
        exit(1)
    os.chdir(cwd)

setup(
    name="pyoperon",
    version="0.3.5",
    description="python bindings for the operon library",
    author='Bogdan Burlacu',
    license="MIT",
    packages=['pyoperon'],
    python_requires=">=3.7",
)
