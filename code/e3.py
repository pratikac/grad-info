import torch as th
import torch.nn as nn
from torch.autograd import Variable

from microbn import MicroBatchNorm2d, MicroBatchNorm1d


opt = dict(b=65536, mbsz=8, mom=0.1, affine=True)
m1 = MicroBatchNorm1d(10, affine=opt['affine'], momentum=opt['mom'], mbsz=opt['mbsz'])
m2 = nn.BatchNorm1d(10, affine=opt['affine'], momentum=opt['mom'])
m2.load_state_dict(m1.state_dict())
m1.train()
m2.train()

x = Variable(th.randn(opt['b'], 10)*10)

for i in xrange(10):
    y1 = m1(x)
    y2 = m2(x)

# print 'g m1: ', m1.weight
# print 'g m2: ', m2.weight
print 'mu m1:', m1.running_mean
print 'mu m2:', m2.running_mean
print 'var m1:', m1.running_var
print 'var m2:', m2.running_var

# print 'mu y1: ', y1.mean(dim=0)
# print 'var y1: ', y1.std(dim=0)**2

# print 'mu y2: ', y2.mean(dim=0)
# print 'var y2: ', y2.std(dim=0)**2