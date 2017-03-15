## Basic FFT ops for pytorch (currently only works on GPU).

### For tensor
````
from torchsignal import fft

# nx2 array represents n complex numbers
x = torch.rand(5, 2).cuda()
y = fft.fft1(x)

# nxhxwx2 array represents n complex images
x = torch.rand(5, 128, 128, 2).cuda()
y = fft.fft2(x)


````

### For Variable
````
from torch.autograd import Variable
from torchsignal import functional as F

# nx2 array represents n complex numbers
x = Variable(torch.rand(5, 2).cuda())
y = F.fft1(x)

# nxhxwx2 array represents n complex images
x = Variable(torch.rand(5, 128, 128, 2).cuda())
y = F.fft2(x)
````
