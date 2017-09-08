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
['-i', '', 'model to load'],
['--check', '', 'check S'],
['--dataset', 'cifar10', 'mnist | cifar10 | cifar100 | svhn | imagenet'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 5, 'max epochs'],
['--lr', 0.1, 'lr'],
['--lrs', '', 'lr schedule'],
['-L', 10, 'stop points for computing S'],
['-s', 42, 'seed'],
['-l', False, 'log'],
['-v', False, 'verbose']
])

setup(opt)

model = getattr(models, opt['m'])(opt).cuda()
if not opt['i'] == '':
    model.load_state_dict(th.load(opt['i'])['state_dict'])
criterion = nn.CrossEntropyLoss().cuda()

build_filename(opt, blacklist=['i', 's', 'check'])
pprint(opt)

dataset, augment = getattr(loader, opt['dataset'])(opt)
loaders = loader.get_loaders(dataset, augment, opt)
data = loaders[0]['train_full']

# populate buffers before flattening params
for bi, (x,y) in enumerate(data):
    x,y = Variable(x.cuda()), Variable(y.cuda())
    model.zero_grad()
    f = criterion(model(x), y)
    f.backward()
    break

N = models.num_parameters(model)
fw, fdw = th.FloatTensor(N).cuda(), th.FloatTensor(N).cuda()
flatten_params(model, fw, fdw)

def full_grad():
    grad = th.FloatTensor(N).cuda().zero_()
    loss, top1, top5 = 0, 0, 0
    n = float(len(data))

    for bi, (x,y) in enumerate(data):
        x,y = Variable(x.cuda()), Variable(y.cuda())
        model.zero_grad()
        yh = model(x)
        f = criterion(model(x), y) + opt['l2']/2.*fw.norm()**2
        f.backward()

        err, err5 = clerr(yh.data, y.data, topk=(1,5))

        loss += f.data[0]
        grad.add_(fdw)
        top1 += err
        top5 += err5

    loss /= n
    grad /= n
    top1 /= n
    top5 /= n
    return loss, grad, top1, top5

fn = opt['check']
if not os.path.isfile(fn):
    S = th.FloatTensor(N, N).zero_()
    ff, fgrad, _, _ = full_grad()

    nb = len(data)
    i = 0
    try:
        for e in xrange(opt['B']):
            for b, (x,y) in enumerate(data):
                x,y = Variable(x.cuda()), Variable(y.cuda())

                model.zero_grad()
                yh = model(x)
                f = criterion(model(x), y) + opt['l2']/2.*fw.norm()**2
                f.backward()

                tmp = fdw.clone().add_(-1, fgrad)
                S.add_(th.ger(tmp, tmp).cpu())
                i += 1

                if b % 100 == 0:
                    print e, b

    except KeyboardInterrupt:
        print 'exiting early...'

    S.div_(i)
    th.save(dict(S=S, fgrad=fgrad), opt['filename']+'.pz')

else:
    print 'Found %s'%fn
    S = th.load(fn)['S'].numpy()
    print np.linalg.matrix_rank(S)


# allcnnt (N=12238)
# cifar10
# b=128, r=1541
# b=256, r=_
# b=512, r=_
# b=1024, r=245
# b=2048, r=125

# cifar100
# b=128, r=1458
# b=256, r=_
# b=512, r=_
# b=1024, r=245
# b=2048, r=