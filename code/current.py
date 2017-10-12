import argparse, math, random
import torch as th
import torch.nn as nn
import torchnet as tnt
import torchvision.transforms as T

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
['-m', 'lenett', 'lenett'],
['-g', 0, 'gpu'],
['--dataset', 'mnist', 'mnist'],
['-b', 128, 'batch_size'],
['-B', 5, 'max epochs'],
['--lr', 0.1, 'lr'],
['--lrs', '', 'lr schedule'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-v', False, 'verbose']
])

N = 100

setup(opt)

model = nn.Sequential(
    nn.Linear(49,16),
    nn.BatchNorm1d(16),
    nn.ReLU(True),
    nn.Linear(16,10)
)
criterion = nn.CrossEntropyLoss().cuda()

build_filename(opt, blacklist=['i', 'check', 'L'])
# pprint(opt)

dataset, augment = getattr(loader, opt['dataset'])(opt)
x, y = dataset['train']['x'], dataset['train']['y']

xs, ys = [], []
for i in xrange(10):
    idx = (y==i).nonzero()[:N//10].squeeze()

    tmp = []
    for ii in idx:
        t = T.ToPILImage()(x[ii].view(1,28,28))
        t = T.Scale(7)(t)
        t = T.ToTensor()(t)
        xs.append(t.view(1,49))
    ys.append(y[idx])
x, y = th.cat(xs), th.cat(ys)
idx = th.randperm(N)
x, y = x[idx], y[idx]
