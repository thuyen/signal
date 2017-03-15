import os
import sys

from setuptools import setup, find_packages

#import torchsignal.build

this_file = os.path.dirname(__file__)

setup(
    name="torchsignal",
    version="0.1",
    description="FFT for pytorch",
    url="https://github.com/thuyen/torchsignal/",
    author="XYZ",
    author_email="vanthuyen@gmail.com",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["torchsignal.build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="torchsignal",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "torchsignal/build.py:ffi")
    ],
)
