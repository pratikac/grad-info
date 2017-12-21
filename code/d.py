import argparse, math, random
import torch as th
import torch.nn as nn
import torchnet as tnt

from torch.autograd import Variable

from exptutils import *
import models, loader
from timeit import default_timer as timer

import numpy as np
import logging
from pprint import pprint
import pdb, glob, sys, gc, time, os
from copy import deepcopy

opt = add_args([
['-o', '/home/%s/local2/pratikac/results'%os.environ['USER'], 'output'],
])

N = 50000
d = 6

model = nn.Sequential(
    models.View(d*d),
    nn.Linear(d*d,10))

# model = models.lenett(opt)
model.eval()
criterion = nn.CrossEntropyLoss()

x = th.FloatTensor(N,1,d,d).normal_()
y = th.LongTensor(N).random_(0,10)

_f = criterion(model(Variable(x[0].view(1,1,d,d))), Variable(th.LongTensor(1)))
_f.backward()
n = models.num_parameters(model)
fw, fdw = flatten_params(model)

def full_grad():
    g = fdw.clone().zero_()
    model.zero_grad()
    _f = criterion(model(Variable(x)), Variable(y))
    _f.backward()
    g.copy_(fdw)
    return g

fg = full_grad()
D = th.FloatTensor(n,n).zero_()
D.add_(-1,th.ger(fg,fg))

for i in xrange(N):
    model.zero_grad()
    _f = criterion(model(Variable(x[i].view(1,1,d,d))), Variable(th.LongTensor(1).fill_(y[i])))
    _f.backward()
    D.add_(1.0/float(N), th.ger(fdw,fdw))

    if i % 100 == 0:
        print i

D = D.numpy()
print 'rank: %f%%'%(np.linalg.matrix_rank(D)/float(n)*100)