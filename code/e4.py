import torch as th
import torch.nn as nn
from torch.autograd import Variable
import pytorch_fft.fft as fft
import numpy as np

a = th.randn(2, 100)
w = th.cumsum(a,1)

wr, wimg = fft.rfft(w.cuda())
p = (wr**2 + wimg**2)**0.5

ff = np.fft.rfft(w.numpy())