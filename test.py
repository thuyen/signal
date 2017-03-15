import numpy as np
import torch
from signal import fft as torch_fft

x = np.array([[1, 2, 3, 4, 5]], dtype='float32')
#x = np.array([[1, 1, 1, 1, 1, 1]], dtype='float32')

s = x.shape
y = np.zeros(s).astype('float32')


X = np.stack([x, y], 2)

Xnp = X.view(np.complex64).reshape(*s)
Ynp = np.fft.fft(Xnp)
print(Ynp.ravel())

Xt = torch.from_numpy(X).cuda()
Yt = torch_fft.fft1(Xt).cpu().numpy()
Yt = Yt.view(np.complex64).reshape(*s)
print(Yt.ravel())
