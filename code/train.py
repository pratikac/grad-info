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
['-g', 0, 'gpu'],
['--dataset', 'cifar10', 'mnist | cifar10 | cifar100 | svhn | imagenet'],
['-d', -1.0, 'dropout'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 5, 'max epochs'],
['--lr', 0.1, 'lr'],
['--lrs', '', 'lr schedule'],
['-L', 5, 'number of ckpts'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-v', False, 'verbose']
])

setup(opt)

model = getattr(models, opt['m'])(opt).cuda()
criterion = nn.CrossEntropyLoss().cuda()

build_filename(opt, blacklist=['i', 'check', 'L'])
pprint(opt)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)
train_data, val_data = loaders[0]['train_full'], loaders[0]['val']

optimizer = th.optim.SGD(model.parameters(), lr=opt['lr'],
            momentum=0.0, weight_decay=opt['l2'])

def train(e):
    opt['lr'] = lrschedule(opt, e)
    for p in optimizer.param_groups:
        p['lr'] = opt['lr']

    model.train()
    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    opt['nb'] = len(train_data)
    for b, (x,y) in enumerate(train_data):
        x,y = Variable(x.cuda()), Variable(y.cuda())
        dt = timer()

        model.zero_grad()
        yh = model(x)
        f = criterion(yh, y)
        f.backward()

        optimizer.step()

        top1.add(yh.data, y.data)
        loss.add(f.item())

        if b % 100 == 0 and b > 0:
            print( '[%03d][%03d/%03d] %.3f %.3f%% [%.3fs]'%(e, b, opt['nb'], \
                    loss.value()[0], top1.value()[0], timer()-dt))

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], train=True)
    print('+[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

def validate(e):
    dry_feed(model, train_data)
    model.eval()

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    nb = len(val_data)
    with th.no_grad():
        for b, (x,y) in enumerate(val_data):
            x,y = Variable(x.cuda()), Variable(y.cuda())

            model.zero_grad()
            yh = model(x)
            f = criterion(yh, y)

            top1.add(yh.data, y.data)
            loss.add(f.item())

    r = dict(e=e, f=loss.value()[0], top1=top1.value()[0], val=True)
    print('*[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

def save_ckpt(e, stats):
    if not opt['l']:
        return

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

# main
tmm, vmm = None, None
for e in range(opt['B']):
    tmm = train(e)

    if e % (opt['B'] // opt['L']) == 0:
        vmm = validate(e)
        save_ckpt(e, dict(train=tmm, val=vmm))

    print()

# save on the last one
vmm = validate(opt['B'])
save_ckpt(opt['B'], dict(train=tmm, val=vmm))
