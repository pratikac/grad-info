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
['-m', 'lenett', 'fclenett | lenett'],
['-g', 0, 'gpu'],
['--dataset', 'mnist', 'mnist'],
['--augment', False, 'augment'],
['-b', 128, 'batch_size'],
['--nc', 5, 'num classes'],
['--frac', 1.0, 'frac'],
['-B', 100000, 'max epochs'],
['--lr', 0.1, 'lr'],
['--l2', 0.0, 'l2'],
['--lrs', '', 'lr schedule'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-v', False, 'verbose']
])

setup(opt)

model = getattr(models, opt['m'])(opt)
if opt['m'] == 'fclenett':
    dataset, augment = loader.halfmnist(opt, 7, nc=opt['nc'])
else:
    dataset, augment = loader.halfmnist(opt, 28, nc=opt['nc'])
loaders = loader.get_loaders(dataset, augment, opt)
train_data = loaders[0]['train']
opt['N'] = int(len(loaders[0]['train_full'])*opt['frac'])

criterion = nn.CrossEntropyLoss()

if opt['g'] >= 0:
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = th.optim.SGD(model.parameters(), lr=opt['lr'],
            momentum=0.9, weight_decay=opt['l2'])

build_filename(opt, blacklist=['i','augment','dataset','m','nc','b','N','d','n'])
pprint(opt)

# dummy populate
for _, (x,y) in enumerate(train_data):
    if opt['g'] >= 0:
        x, y = x.cuda(), y.cuda()
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

    opt['nb'] = len(loaders[0]['train_full'])*opt['frac']
    for b, (x,y) in enumerate(train_data):
        if opt['g'] >= 0:
            x, y = x.cuda(), y.cuda()
        x,y = Variable(x), Variable(y)

        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)
        f.backward()

        optimizer.step()

        top1.add(yh.data, y.data)
        loss.add(f.data[0])

        if b > opt['nb']:
            break

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], train=True)
    print '+[%02d] %.3f %.3f%% %.2fs\n'%(e, r['f'], r['top1'], timer()-dt)
    return r

def full_grad():
    model.train()
    dwc = dw.clone().zero_()

    opt['nb'] = len(loaders[0]['train_full'])
    for b, (x,y) in enumerate(loaders[0]['train_full']):
        if opt['g'] >= 0:
            x, y = x.cuda(), y.cuda()
        x,y = Variable(x), Variable(y)

        model.zero_grad()
        f = criterion(model(x), y)
        f.backward()
        dwc.add_(dw)

    return dwc/float(opt['nb'])

fs, top1s = [], []
ws, dws, moms = [], [], []
try:
    for e in xrange(opt['B']):
        r = train()

        if e > 1000:
            fs.append(r['f'])
            top1s.append(r['top1'])
            ws.append(w.clone().cpu())
            dws.append(full_grad().cpu())

            mom = th.cat([optimizer.state[p]['momentum_buffer'].view(-1) for p in model.parameters()])
            moms.append(mom.clone().cpu())
except KeyboardInterrupt:
    pass

if opt['l']:
    print 'Saving...'
    r = gitrev(opt)
    th.save(dict(w=th.cat(ws).view(-1,opt['np']).t().numpy(), dw=th.cat(dws).view(-1,opt['np']).t().numpy(),
                mom=th.cat(moms).view(-1,opt['np']).t().numpy(),
                opt=opt,
                meta=dict(SHA=r[0], STATUS=r[1], DIFF=r[2]),
                f=fs,top1=top1s), opt['filename'] + '_trajectory.pz')