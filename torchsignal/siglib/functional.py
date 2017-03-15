from torch.autograd import Function
from . import lib_fft

class FFT1(Function):
    def forward(self, input):
        # [... n, 2]
        size = input.size()
        input = input.view(-1, *size[-2:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft1_c2c(input, output, 1)
        else:
            lib_fft.fft1_c2c_cuda(input, output, 1)
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-2:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft1_c2c(grad_output, grad_input, -1)
        else:
            lib_fft.fft1_c2c_cuda(grad_output, grad_input, -1)
        return grad_input.view(size)

class FFT2(Function):
    def forward(self, input):
        # size [... h, w, 2]
        size = input.size()
        input = input.view(-1, *size[-3:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft2_c2c(input, output, 1)
        else:
            lib_fft.fft2_c2c_cuda(input, output, 1)
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-3:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft2_c2c(grad_output, grad_input, -1)
        else:
            lib_fft.fft2_c2c_cuda(grad_output, grad_input, -1)
        return grad_input.view(size)

def fft1(input):
    f = FFT1()
    return f(input)

def fft2(input):
    f = FFT2()
    return f(input)
