from torch.autograd import Function
from . import lib_fft

class FFT1(Function):
    def forward(self, input):
        # [..., n, 2]
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

class iFFT1(Function):
    def forward(self, input):
        # [..., n, 2]
        size = input.size()
        input = input.view(-1, *size[-2:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft1_c2c(input, output, -1)
        else:
            lib_fft.fft1_c2c_cuda(input, output, -1)
        output.mul_(size[-2])
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-2:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft1_c2c(grad_output, grad_input, 1)
        else:
            lib_fft.fft1_c2c_cuda(grad_output, grad_input, 1)
        grad_input.div_(size[-2])
        return grad_input.view(size)

class FFT2(Function):
    def forward(self, input):
        # size [..., h, w, 2]
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

class iFFT2(Function):
    def forward(self, input):
        # size [..., h, w, 2]
        size = input.size()
        input = input.view(-1, *size[-3:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft2_c2c(input, output, -1)
        else:
            lib_fft.fft2_c2c_cuda(input, output, -1)
        output.div_(size[-2]*size[-3])
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-3:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft2_c2c(grad_output, grad_input, 1)
        else:
            lib_fft.fft2_c2c_cuda(grad_output, grad_input, 1)
        grad_input.div_(size[-2]*size[-3])
        return grad_input.view(size)

class FFT3(Function):
    def forward(self, input):
        # size [..., h, w, t, 2]
        size = input.size()
        input = input.view(-1, *size[-4:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft3_c2c(input, output, 1)
        else:
            lib_fft.fft3_c2c_cuda(input, output, 1)
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-4:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft3_c2c(grad_output, grad_input, -1)
        else:
            lib_fft.fft3_c2c_cuda(grad_output, grad_input, -1)
        return grad_input.view(size)

class iFFT3(Function):
    def forward(self, input):
        # size [..., h, w, t, 2]
        size = input.size()
        input = input.view(-1, *size[-4:])
        output = input.new()
        if not input.is_cuda:
            lib_fft.fft3_c2c(input, output, -1)
        else:
            lib_fft.fft3_c2c_cuda(input, output, -1)
        output.div_(size[-2]*size[-3]*size[-4])
        return output.view(size)

    def backward(self, grad_output):
        size = grad_output.size()
        grad_output = grad_output.view(-1, *size[-4:])
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            lib_fft.fft3_c2c(grad_output, grad_input, 1)
        else:
            lib_fft.fft3_c2c_cuda(grad_output, grad_input, 1)
        grad_input.div_(size[-2]*size[-3]*size[-4])
        return grad_input.view(size)

def fft1(input):
    f = FFT1()
    return f(input)

def fft2(input):
    f = FFT2()
    return f(input)

def fft3(input):
    f = FFT3()
    return f(input)

def ifft1(input):
    f = iFFT1()
    return f(input)

def ifft2(input):
    f = iFFT2()
    return f(input)

def ifft3(input):
    f = iFFT3()
    return f(input)
