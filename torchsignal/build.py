import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/fft.c']
headers = ['src/fft.h']
defines = []
with_cuda = False
libraries = ['fftw3f']

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/fft_cuda.c']
    headers += ['src/fft_cuda.h']
    defines += [('WITH_CUDA', None)]
    libraries += ['cufft']
    with_cuda = True

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
