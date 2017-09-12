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
['-m', 'allcnnt', 'lenet | mnistfc | allcnn | wrn* | resnet*'],
['-r', '', 'retrain'],
['-g', 0, 'gpu'],
['--dataset', 'cifar10', 'mnist | cifar10 | cifar100 | svhn | imagenet'],
['-d', -1.0, 'dropout'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 50, 'max epochs'],
['--lr', 0.1, 'lr'],
['--lrs', '', 'lr schedule'],
['-L', 5, 'number of ckpts'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['--no_microbn', False, 'no microbn'],
['--burnin', 5, 'burnin'],
['--no_dual_batch', False, 'no dual_batch'],
['-v', False, 'verbose']
])

best_mm = None

setup(opt)

model = getattr(models, opt['m'])(opt, microbn=not opt['no_microbn']).cuda()
if opt['g'] >= th.cuda.device_count():
    print '[Using DataParallel]'
    model = nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss().cuda()

build_filename(opt, blacklist=['i', 'check', 'r', 'no_microbn', 'no_dual_batch', 'burnin'])
pprint(opt)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)
train_data, val_data = loaders[0]['train_full'], loaders[0]['val']

_opt = deepcopy(opt)
_opt['b'] = 128
loaders_128 = loader.get_loaders(dataset, augment, _opt)
train_data_128, val_data_128 = loaders_128[0]['train_full'], loaders_128[0]['val']

optimizer = th.optim.SGD(model.parameters(), lr=opt['lr'],
            momentum=0.9, weight_decay=opt['l2'])

def train(e):
    opt['lr'] = lrschedule(opt, e)

    model.train()
    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    def step(x,y, log=True, lr=None, step=True):
        x,y = Variable(x.cuda()), Variable(y.cuda())
        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)
        f.backward()

        if lr is None:
            optimizer.param_groups[0]['lr'] = min(opt['lr']*x.size(0)/128.0, 1)
            optimizer.param_groups[0]['momentum'] = 0.9
        else:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[0]['momentum'] = 0

        if step:
            optimizer.step()

        if log:
            top1.add(yh.data, y.data)
            loss.add(f.data[0])

    dt = timer()
    if e < opt['burnin']:
        opt['nb'] = len(train_data_128)
        for b, (x,y) in enumerate(train_data_128):
            _dt = timer()
            step(x,y)

            if b % 50 == 0 and b > 0:
                print '[%03d][%03d/%03d] %.4f %.3f%% [%.3fs]'%(e, b, opt['nb'], \
                        loss.value()[0], top1.value()[0], timer()-_dt)
    else:
        b = 0
        opt['nb'] = len(train_data)
        for (x,y), (x128, y128) in zip(train_data, train_data_128):
            _dt = timer()
            if not opt['no_dual_batch']:
                step(x128, y128, log=False, lr=opt['lr'])
                step(x128, y128, log=False, lr=-opt['lr'])
            step(x,y)
            b += 1

            if b % 5 == 0 and b > 0:
                print '[%03d][%03d/%03d] %.4f %.3f%% [%.3fs]'%(e, b, opt['nb'], \
                        loss.value()[0], top1.value()[0], timer()-_dt)

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], train=True)
    print '+[%02d] %.4f %.3f%% [%.3fs]'%(e, r['f'], r['top1'], timer()-dt)
    return r

def validate(e):
    dry_feed(model, train_data)
    model.eval()

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    nb = len(val_data)
    for b, (x,y) in enumerate(val_data):
        x,y = Variable(x.cuda(), volatile=True), Variable(y.cuda(), volatile=True)

        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)

        top1.add(yh.data, y.data)
        loss.add(f.data[0])

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], val=True)
    print '*[%02d] %.4f %.3f%%'%(e, r['f'], r['top1'])
    return r

def save_ckpt(e, stats):
    if not opt['l']:
        return

    global best_mm

    loc = opt.get('o')
    dirloc = os.path.join(loc, opt['m'], opt['filename'])
    if not os.path.isdir(dirloc):
        os.makedirs(dirloc)

    fn = '%s_%02d.pz'%(opt['m'], e)
    r = gitrev(opt)
    meta = dict(SHA=r[0], STATUS=r[1], DIFF=r[2])
    th.save(dict(
            meta = meta,
            opt=json.dumps(opt),
            state_dict=model.state_dict(),
            stats=stats,
            e=e),
            os.path.join(dirloc, fn))

def load_ckpt():
    if opt['r'] == '':
        return

    print('Loading model from: ', opt['r'])
    d = th.load(opt['r'])
    model.load_state_dict(d['state_dict'])
    opt['e'] = d['e'] + 1
    print('[Loaded model, check validation error]')
    validate(d['e'])

# main
opt['e'] = 0
load_ckpt()
tmm, vmm = None, None
for e in xrange(opt['e'], opt['B']):
    tmm = train(e)
    vmm = validate(e)
    save_ckpt(e, dict(train=tmm, val=vmm))

    print ''

# save on the last one
vmm = validate(opt['B'])
save_ckpt(opt['B'], dict(train=tmm, val=vmm))