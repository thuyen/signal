import os
import sys

from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

setup(
    name="torchsignal",
    version="0.1",
    description="Signal Processing for Pytorch",
    url="https://github.com/thuyen/signal/",
    author="Thuyen Ngo",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
