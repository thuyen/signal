from . import lib_fft

def fft1(input):
    # [..., n, 2]
    size = input.size()
    input = input.view(-1, *size[-2:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft1_c2c(input, output, 1)
    else:
        lib_fft.fft1_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft1(input):
    # size [..., n, 2]
    size = input.size()
    input = input.view(-1, *size[-2:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft1_c2c(input, output, -1)
    else:
        lib_fft.fft1_c2c_cuda(input, output, -1)
    output.div_(size[-2])
    return output.view(size)

def fft2(input):
    # size [..., w, h, 2]
    size = input.size()
    input = input.view(-1, *size[-3:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft2_c2c(input, output, 1)
    else:
        lib_fft.fft2_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft2(input):
    # size [..., w, h, 2]
    size = input.size()
    input = input.view(-1, *size[-3:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft2_c2c_cuda(input, output, -1)
    else:
        lib_fft.fft2_c2c_cuda(input, output, -1)
    output.div_(size[-3])
    return output.view(size)

def fft3(input):
    # size [..., w, h, t, 2]
    size = input.size()
    input = input.view(-1, *size[-4:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft3_c2c(input, output, 1)
    else:
        lib_fft.fft3_c2c_cuda(input, output, 1)
    return output.view(size)

def ifft3(input):
    # size [..., w, h, t, 2]
    size = input.size()
    input = input.view(-1, *size[-4:])
    output = input.new()
    if not input.is_cuda:
        lib_fft.fft3_c2c_cuda(input, output, -1)
    else:
        lib_fft.fft3_c2c_cuda(input, output, -1)
    output.div_(size[-3])
    return output.view(size)
