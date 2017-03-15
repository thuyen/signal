## Basic FFT ops for pytorch (with autograd ops) on GPU

### For Variable
````
from torch.autograd import Variable
from torchsignal import functional as F

# nx2 represent n complex numbers
x = Variable(torch.rand(5, 2).cuda())
y = F.fft1()(x)
````

### For tensor
````
from torchsignal import fft

# nx2 represent n complex numbers
x = torch.rand(5, 2).cuda()
y = fft.fft1(x)
````
