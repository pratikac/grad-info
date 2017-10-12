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
['--augment', False, 'augment'],
['-b', 128, 'batch_size'],
['--nc', 4, 'num classes'],
['-B', 100000, 'max epochs'],
['--lr', 0.1, 'lr'],
['--l2', 0.0, 'l2'],
['--lrs', '', 'lr schedule'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-v', False, 'verbose']
])

setup(opt)

c = 16
opt['N'] = 512*opt['nc']
model = nn.Sequential(
    nn.Linear(49,c),
    nn.BatchNorm1d(c),
    nn.ReLU(True),
    nn.Linear(c,opt['nc'])
)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.SGD(model.parameters(), lr=opt['lr'],
            momentum=0.9, weight_decay=opt['l2'])

build_filename(opt, blacklist=['i','augment','dataset','m','N','nc','b'])
pprint(opt)

def get_data():
    dataset, augment = getattr(loader, opt['dataset'])(opt)
    x, y = dataset['train']['x'], dataset['train']['y']

    xs, ys = [], []
    for i in xrange(opt['nc']):
        idx = (y==i).nonzero()[:opt['N']//opt['nc']].squeeze()

        tmp = []
        for ii in idx:
            t = T.ToPILImage()(x[ii].view(1,28,28))
            t = T.Scale(7)(t)
            t = T.ToTensor()(t)
            xs.append(t.view(1,49))
        ys.append(y[idx])
    x, y = th.cat(xs), th.cat(ys)
    idx = th.randperm(opt['N'])

    dataset = dict(train={'x': x, 'y': y}, val={'x': x, 'y': y})
    loaders = loader.get_loaders(dataset, augment, opt)
    train_data, val_data = loaders[0]['train_full'], loaders[0]['val']
    return train_data

train_data = get_data()

# dummy populate
for _, (x,y) in enumerate(train_data):
    _f = criterion(model(Variable(x)), Variable(y))
    _f.backward()
    break
w, dw = flatten_params(model)
opt['np'] = w.numel()
print 'Num parameters: ', opt['np']

def train():
    dt = timer()

    opt['lr'] = lrschedule(opt, e)
    for p in optimizer.param_groups:
        p['lr'] = opt['lr']

    model.train()
    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    opt['nb'] = len(train_data)
    for b, (x,y) in enumerate(train_data):
        x,y = Variable(x), Variable(y)

        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)
        f.backward()

        optimizer.step()

        top1.add(yh.data, y.data)
        loss.add(f.data[0])

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], train=True)
    print '+[%02d] %.3f %.3f%% %.2fs\n'%(e, r['f'], r['top1'], timer()-dt)
    return r

fs, top1s, ws, dws = [], [], [], []
try:
    for e in xrange(opt['B']):
        r = train()

        if e > 200:
            fs.append(r['f'])
            top1s.append(r['top1'])
            ws.append(w.clone())
            dws.append(dw.clone())
except KeyboardInterrupt:
    pass

if opt['l']:
    print 'Saving...'
    th.save(dict(w=th.cat(ws).view(-1,opt['np']), dw=th.cat(dws).view(-1,opt['np']),
                f=fs,top1=top1s), opt['filename'] + '_trajectory.pz')