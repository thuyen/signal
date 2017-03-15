from torch.autograd import Function
from . import lib_fft
import numpy as np
import torch

class FFT1(Function):
    def forward(self, input):
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

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-2:])
        if not grad_output.is_cuda:
            grad_output = grad_output.numpy().view(np.complex64)
            grad_input = np.fft.ifft(grad_output, axis=-2)
            grad_input = np.ascontiguousarray(grad_input.astype(np.complex64))
            grad_input = torch.from_numpy(grad_input.view(np.float32))
            grad_input.mul_(size[-2])
        else:
            grad_input = grad_output.new()
            lib_fft.fft1_c2c_cuda(grad_output, grad_input, -1)
        return grad_input.view(size)

class FFT2(Function):
    def forward(self, input):
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

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-3:])
        if not grad_output.is_cuda:
            grad_output = grad_output.numpy().view(np.complex64)
            grad_input = np.fft.ifft2(grad_output, axes=(-3, -2))
            grad_input = np.ascontiguousarray(grad_input.astype(np.complex64))
            grad_input = torch.from_numpy(grad_input.view(np.float32))
            grad_input.mul_(size[-2]*size[-3])
        else:
            grad_input = grad_output.new()
            lib_fft.fft2_c2c_cuda(grad_output, grad_input, -1)
        return grad_input.view(size)

def fft1(input):
    f = FFT1()
    return f(input)

def fft2(input):
    f = FFT2()
    return f(input)
