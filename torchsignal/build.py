import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/fft_cuda.c']
headers = ['src/fft_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True
libraries = ['cufft']

ffi = create_extension(
    'siglib.lib_fft',
    headers=headers,
    package=True,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    libraries=libraries,
    with_cuda=with_cuda
)


if __name__ == '__main__':
    ffi.build()
