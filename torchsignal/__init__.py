import torch
from siglib import fft, functional

def fft1(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.fft1(input)
    else:
        return fft.fft1(input)


def fft2(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.fft2(input)
    else:
        return fft.fft2(input)

def fft3(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.fft3(input)
    else:
        return fft.fft3(input)

def ifft1(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.ifft1(input)
    else:
        return fft.ifft1(input)

def ifft2(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.ifft2(input)
    else:
        return fft.ifft2(input)

def ifft3(input):
    if isinstance(input, torch.autograd.Variable):
        return functional.ifft3(input)
    else:
        return fft.ifft3(input)
