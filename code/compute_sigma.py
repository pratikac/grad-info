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
import hickle as hkl

opt = add_args([
['-g', 0, 'gpu'],
['-i', '', 'location of ckpts'],
['-b', 1, 'batch_size'],
['--augment', False, 'data augmentation'],
['-B', 5, 'max epochs'],
['-l', False, 'log'],
['--stats', False, 'compute stats']
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
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    dataset, augment = getattr(loader, opt['dataset'])(opt)
    loaders = loader.get_loaders(dataset, augment, opt)
    data = loaders[0]['train_full']

    # populate buffers before flattening params
    for bi, (x,y) in enumerate(data):
        _f = criterion(model(Variable(x.cuda())), Variable(y.cuda()))
        _f.backward()
        break

    N = models.num_parameters(model)
    fw, fdw = flatten_params(model)

    def full_grad():
        _opt = deepcopy(opt)
        _opt['b'] = 1024
        _full_loaders = loader.get_loaders(dataset, augment, _opt)
        _data = _full_loaders[0]['train_full']

        grad = th.FloatTensor(N).cuda().zero_()

        for _, (x,y) in enumerate(_data):
            x,y = Variable(x.cuda()), Variable(y.cuda())
            model.zero_grad()
            yh = model(x)
            _f = criterion(yh, y) + opt['l2']/2.*fw.norm()**2
            _f.backward()

            grad.add_(fdw)

        grad.div_(len(_data))
        return grad

    print 'S: ', f+'.S_augment_%s.pz'%(opt['augment'])

    print '[computing full grad]'
    fgrad = full_grad()

    print '[computing S]'
    S = th.FloatTensor(N,N).zero_().cuda()

    # with bsz = 1
    opt['nb'] = len(data)
    for b, (x,y) in enumerate(data):
        _dt = timer()
        x,y = Variable(x.cuda()), Variable(y.cuda())
        model.zero_grad()
        yh = model(x)
        _f = criterion(yh, y) + opt['l2']/2.*fw.norm()**2
        _f.backward()

        tmp = fdw.clone().add_(-1, fgrad)
        S.add_(th.ger(tmp,tmp))

        if b % 100 == 0:
            print '[%d] %.2fs'%(b, timer()-_dt)

    S.div_(opt['nb'])
    fn = f+'.S_augment_%s.pz'%(opt['augment'])
    res = dict(opt=opt, S=S.cpu(), fgrad=fgrad.cpu())
    th.save(res, fn)

def compute_stats(f):
    fn = f+'.S.pz'
    if not os.path.isfile(fn):
        print 'File %s not found, runing helper()'%f+'.S.pz'
        helper(f)

    d = th.load(fn)
    S, fgrad = d['S'].numpy(), d['fgrad'].numpy()
    S2 = np.outer(fgrad, fgrad)
    S1 = S + S2

    print '[begin eig]...'
    dt = timer()
    eig, evec = np.linalg.eigh(S)
    eig, evec = eig.real, evec.real
    print '[finished eig]... ', timer()-dt

    sval = eig
    eps = sval.max() * S.shape[0] * np.finfo(np.float32).eps
    rank = (sval > eps).sum()

    n = 1000
    res = dict(eig=eig, evec=evec[:,:n], sval=sval, rank=rank)
    th.save(res, f+'.eig.pz')

if __name__ == '__main__':
    if '*' in opt['i']:
        print 'Found files: '
        for f in sorted(glob2.glob(opt['i'] + '/*.pz')):
            print f

        print 'continue?'
        raw_input()
        for f in sorted(glob2.glob(opt['i'] + '/*.pz')):
            if opt['stats']:
                compute_stats(f)
            else:
                helper(f)
    else:
        if opt['stats']:
            compute_stats(opt['i'])
        else:
            helper(opt['i'])