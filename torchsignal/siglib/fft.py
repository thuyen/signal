from . import lib_fft
import numpy as np
import torch

def fft1(input):
    # [... n, 2]
    size = input.size()
    input = input.view(-1, *size[-2:])
    if not input.is_cuda:
        input = input.numpy().view(np.complex64)
        output = np.fft.fft(input, axis=-2)
        output = np.ascontiguousarray(output.astype(np.complex64))
        output = torch.from_numpy(output.view(np.float32))
    else:
        output = input.new()
        lib_fft.fft1_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft1(input):
    # size [... n, 2]
    size = input.size()
    input = input.view(-1, *size[-2:])
    if not input.is_cuda:
        input = input.numpy().view(np.complex64)
        output = np.fft.ifft(input, axis=-2)
        output = np.ascontiguousarray(output.astype(np.complex64))
        output = torch.from_numpy(output.view(np.float32))
    else:
        output = input.new()
        lib_fft.fft1_c2c_cuda(input, output, -1)
        output.div_(size[-2])
    return output.view(size)

def fft2(input):
    # size [... h, w, 2]
    size = input.size()
    input = input.view(-1, *size[-3:])
    if not input.is_cuda:
        input = input.numpy().view(np.complex64)
        output = np.fft.fft2(input, axes=(-3, -2))
        output = np.ascontiguousarray(output.astype(np.complex64))
        output = torch.from_numpy(output.view(np.float32))
    else:
        output = input.new()
        lib_fft.fft2_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft2(input):
    # size [... h, w, 2]
    size = input.size()
    input = input.view(-1, *size[-3:])
    if not input.is_cuda:
        input = input.numpy().view(np.complex64)
        output = np.fft.ifft2(input, axes=(-3, -2))
        output = np.ascontiguousarray(output.astype(np.complex64))
        output = torch.from_numpy(output.view(np.float32))
    else:
        output = input.new()
        lib_fft.fft2_c2c_cuda(input, output, -1)
        output.div_(size[-2]*size[-3])
    return output.view(size)
