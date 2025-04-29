from setuptools import setup, find_packages
import os

# Read version from cubexplore/__init__.py
def read_version():
    version_file = os.path.join("cubexplore", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]

setup(
    name='cubexplore',
    version=read_version(),
    packages=find_packages(),
    install_requires=[
      
    ],  # or parse from requirements.txt
    description='Exploration tools for hyperspectral cubes',
    author='chilly-nk',
)