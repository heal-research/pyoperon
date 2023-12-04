from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="pyoperon",
    version="0.3.6",
    description="python bindings for the operon library",
    author='Bogdan Burlacu',
    packages=['pyoperon'],
    python_requires=">=3.8",
)
