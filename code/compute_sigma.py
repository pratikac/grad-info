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
import pdb, glob2, sys, gc, time, os, json
from copy import deepcopy

opt = add_args([
['-g', 0, 'gpu'],
['-i', '', 'location of modules'],
['-b', 128, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 5, 'max epochs'],
['-l', False, 'log'],
])

setup(opt)
optc = deepcopy(opt)

def helper(f):
    print '[Processing] ', f
    ckpt = th.load(f)
    _opt = json.loads(ckpt['opt'])

    _opt.update(**optc)
    opt = deepcopy(_opt)
    pprint(opt)

    model = getattr(models, opt['m'])(opt).cuda()
    model.load_state_dict(ckpt['state_dict'])
    criterion = nn.CrossEntropyLoss().cuda()

    dataset, augment = getattr(loader, opt['dataset'])(opt)
    loaders = loader.get_loaders(dataset, augment, opt)
    data = loaders[0]['train_full']
    opt['nb'] = len(data)

    # populate buffers before flattening params
    for bi, (x,y) in enumerate(data):
        x,y = Variable(x.cuda()), Variable(y.cuda())
        model.zero_grad()
        _f = criterion(model(x), y)
        _f.backward()
        break

    N = models.num_parameters(model)
    fw, fdw = th.FloatTensor(N).cuda(), th.FloatTensor(N).cuda()
    flatten_params(model, fw, fdw)
    model.train()

    def full_grad():
        grad = th.FloatTensor(N).cuda().zero_()

        for bi, (x,y) in enumerate(data):
            x,y = Variable(x.cuda()), Variable(y.cuda())
            model.zero_grad()
            yh = model(x)
            _f = criterion(model(x), y) + opt['l2']/2.*fw.norm()**2
            _f.backward()
            grad.add_(fdw)

        grad.div_(opt['nb'])
        return grad

    print '[start computing S]'
    S = th.FloatTensor(N,N).zero_().cuda()
    fgrad = full_grad()

    i = 0
    dt = timer()
    for e in xrange(opt['B']):
        for b, (x,y) in enumerate(data):
            _dt = timer()
            x,y = Variable(x.cuda()), Variable(y.cuda())

            model.zero_grad()
            yh = model(x)
            _f = criterion(model(x), y) + opt['l2']/2.*fw.norm()**2
            _f.backward()

            tmp = fdw.clone().add_(-1, fgrad)
            S.add_(th.ger(tmp, tmp))
            i += 1

            if b % 10 == 0:
                print e, b, timer()-_dt

    S.div_(i)
    S = S.cpu().numpy()

    print '[finished computing S]... ', timer()-dt
    print '[begin eig]...'
    dt = timer()
    eig = np.linalg.eigvalsh(S)
    print '[finished eig]... ', timer()-dt
    print 'eig: ', eig[:100]

    print '[begin svd]...'
    dt = timer()
    sval = np.linalg.svd(S, compute_uv=False)
    print '[finished svd]... ', timer()-dt
    print 'sval: ', sval[:100]

    eps = sval.max() * N * np.finfo(np.float32).eps
    print 'eps: ', eps
    rank = (sval > eps).sum()
    print 'rank: ', rank

    print '[save back ckpt]...'
    if not 'b' in ckpt:
        ckpt['b'] = {}
    if opt['b'] in ckpt['b']:
        print 'found key: ', opt['b'], ' in ckpt[b], will overwrite'
    ckpt['b'][opt['b']] = dict(eig=eig, sval=sval, rank=rank)

    _ckpt = dict(eig=eig, sval=sval, rank=rank)

    th.save(_ckpt, f+'.eig.pz')
    th.save(ckpt, f)

if '*' in opt['i']:
    print 'Found files: '
    for f in sorted(glob2.glob(opt['i'] + '/*.pz')):
        print f

    print 'continue?'
    raw_input()
    for f in sorted(glob2.glob(opt['i'] + '/*.pz')):
        helper(f)
else:
    helper(opt['i'])
