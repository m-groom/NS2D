"""
Setup script for NS2D package.

This allows the package to be installed in development mode:
    pip install -e .

Or for regular installation:
    pip install .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ns2d",
    version="0.1.0",
    author="Michael Groom",
    description="2D Incompressible Navier-Stokes Solver using Dedalus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m-groom/NS2D",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
