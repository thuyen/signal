from . import lib_fft

def fft1(input):
    # [... a, 2]
    if not input.is_cuda:
        raise ValueError('Only GPU tensors are supported')
    size = input.size()
    input = input.view(-1, *size[-2:])
    output = input.new()
    lib_fft.fft1_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft1(input):
    # size [... a, 2]
    if not input.is_cuda:
        raise ValueError('Only GPU tensors are supported')
    size = input.size()
    input = input.view(-1, *size[-2:])
    output = input.new()
    lib_fft.fft1_c2c_cuda(input, output, -1)
    output.div_(size[-2])
    return output.view(size)

def fft2(input):
    # size [... a, b, 2]
    if not input.is_cuda:
        raise ValueError('Only GPU tensors are supported')
    size = input.size()
    input = input.view(-1, *size[-3:])
    output = input.new()
    lib_fft.fft2_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft2(input):
    # size [... a, b, 2]
    if not input.is_cuda:
        raise ValueError('Only GPU tensors are supported')
    size = input.size()
    input = input.view(-1, *size[-3:])
    output = input.new()
    lib_fft.fft2_c2c_cuda(input, output, -1)
    output.div_(size[-3])
    return output.view(size)
