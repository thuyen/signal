## Basic Signal Processing (FFTs) for PyTorch

### For Tensors
````
from torchsignal import fft

# [..., n, 2] array represents complex numbers
x = torch.rand(5, 2).cuda()
y = fft.fft1(x)

# [..., h, w, 2] array represents complex images
x = torch.rand(5, 128, 128, 2).cuda()
y = fft.fft2(x)
````

### For Variables
````
from torch.autograd import Variable
from torchsignal import functional as F

x = Variable(torch.rand(5, 2).cuda())
y = F.fft1(x)

x = Variable(torch.rand(5, 128, 128, 2).cuda())
y = F.fft2(x)
````
