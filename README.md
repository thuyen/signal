## Basic Signal Processing (FFTs) for PyTorch

### Supported OPs
`fft1`, `fft2`, `fft3`, `ifft1`, `ifft2`, `ifft3`.

### Examples
````
import torch
from torch.autograd import Variable
from torchsignal import fft1, fft2

# [..., n, 2] array represents complex numbers
x = torch.rand(5, 2).cuda()
y = fft1(x)

# [..., h, w, 2] array represents complex images
x = torch.rand(5, 128, 128, 2).cuda()
y = fft2(x)

# For variables
x = Variable(torch.rand(5, 2).cuda())
y = fft1(x)

x = Variable(torch.rand(5, 128, 128, 2).cuda())
y = fft2(x)
````
