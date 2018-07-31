import torch as th
import torchvision.transforms as T
import cvtransforms as cv
from torchvision import datasets
import torchnet as tnt
import torch.utils.data
import torchnet as tnt

import numpy as np
import os, sys, pdb, math, random
import cv2
import scipy.io as sio

home = '/home/'+os.environ['USER']

class InfDS(object):
    def __init__(self, d):
        self.d = d
        self.n = d['x'].size(0)

    def __getitem__(self, idx):
        i = idx % self.n
        return [self.d['x'][i], self.d['y'][i]]

    def __len__(self):
        return 2**20

def get_inf_iterator(d, transforms, bsz, nw=0, shuffle=True, pin_memory=True):
    ds = InfDS(d)
    ds = tnt.dataset.TransformDataset(ds, {0:transforms})
    return th.utils.data.DataLoader(ds, batch_size=bsz,
            num_workers=nw, shuffle=shuffle, pin_memory=pin_memory)

def get_iterator(d, transforms, bsz, nw=0, shuffle=True):
    ds = tnt.dataset.TensorDataset([d['x'], d['y']])
    ds = ds.transform({0:transforms})
    return ds.parallel(batch_size=bsz,
            num_workers=nw, shuffle=shuffle, pin_memory=True)

def shuffle_data(d):
    x, y = d['x'], d['y']
    n = x.size(0)
    idx = th.randperm(n)
    d['x'] = th.index_select(x, 0, idx)
    d['y'] = th.index_select(y, 0, idx)

def get_loaders(d, transforms, opt):
    if not opt['augment']:
        transforms = lambda x: x

    opt['frac'] = opt.get('frac', 1.0)
    opt['nw'] = opt.get('nw', 0)
    opt['n'] = opt.get('n', 1)

    trf = get_iterator(d['train'], transforms, opt['b'], nw=opt['nw'], shuffle=True)
    trinff = get_inf_iterator(d['train'], transforms, opt['b'], nw=opt['nw'], shuffle=True)
    tv = get_iterator(d['val'], lambda x:x, opt['b'], nw=opt['nw'], shuffle=False)

    if opt['frac'] > 1-1e-12:
        return [dict(train=trinff,val=tv,test=tv,train_full=trf,
                idx=th.arange(0,d['train']['x'].size(0))) for i in range(opt['n'])]
    else:
        n = opt['n']
        N = d['train']['x'].size(0)
        tr = []
        idxs = []
        for i in range(n):
            fs = (i / float(n)) % 1.0
            ns, ne = int(N*fs), int(N*(fs+opt['frac']))
            x, y = d['train']['x'], d['train']['y']

            if ne <= N:
                idxs.append(th.arange(ns,ne).long())
                xy = {'x': x[ns:ne], 'y': y[ns:ne]}
            else:
                ne = ne % N
                idxs.append(th.cat((th.arange(ns,N), th.arange(0,ne))).long())
                xy = {  'x': th.cat((x[ns:], x[:ne])),
                        'y': th.cat((y[ns:], y[:ne]))}
            tr.append(get_inf_iterator(xy, transforms, opt['b'], nw=0, shuffle=True))
        return [dict(train=tr[i],val=tv,test=tv,train_full=trf,idx=idxs[i]) for i in range(opt['n'])]

def halfmnist(opt, sz=7, nc=2):
    loc = home + '/local2/pratikac/mnist'
    d1, d2 = datasets.MNIST(loc, train=True), datasets.MNIST(loc, train=False)

    d = {'train': {'x': d1.train_data.view(-1,1,28,28).float(), 'y': d1.train_labels},
        'val': {'x': d2.test_data.view(-1,1,28,28).float(), 'y': d2.test_labels}}
    shuffle_data(d['train'])

    idx = d['train']['y'].numpy() < nc
    d['train']['x'] = th.from_numpy(d['train']['x'].numpy()[idx])
    d['train']['y'] = d['train']['y'][d['train']['y'] < nc]

    idx = d['val']['y'].numpy() < nc
    d['val']['x'] = th.from_numpy(d['val']['x'].numpy()[idx])
    d['val']['y'] = d['val']['y'][d['val']['y'] < nc]

    txs, vxs = [], []

    _txs = d['train']['x']
    for i in range(len(_txs)):
        t = T.ToPILImage()(_txs[i])
        t = T.Scale(sz)(t)
        txs.append(T.ToTensor()(t).view(-1,1,sz,sz))
    d['train']['x'] = th.cat(txs)

    _vxs = d['val']['x']
    for i in range(len(_vxs)):
        t = T.ToPILImage()(_vxs[i])
        t = T.Scale(sz)(t)
        vxs.append(T.ToTensor()(t).view(-1,1,sz,sz))
    d['val']['x'] = th.cat(vxs)

    return d, lambda x: x

def mnist(opt):
    loc = home + '/local2/pratikac/mnist'
    d1, d2 = datasets.MNIST(loc, train=True), datasets.MNIST(loc, train=False)

    d = {'train': {'x': d1.train_data.view(-1,1,28,28).float(), 'y': d1.train_labels},
        'val': {'x': d2.test_data.view(-1,1,28,28).float(), 'y': d2.test_labels}}

    shuffle_data(d['train'])
    return d, lambda x: x

def cifar_helper(opt, s):
    if 'cifar' in s:
        loc = home + '/local2/pratikac/cifar/'
    elif 'imagenet32' in s:
        loc = home + '/local2/pratikac/imagenet32/'

    # csz = 16 if opt['dataset'] == 'cifar10' else 8
    # cutout = cv.CutOut(csz, (0,0,0))

    if 'resnet' in opt['m'] or 'densenet' in opt['m']:
        d1 = np.load(loc+s+'-train.npz')
        d2 = np.load(loc+s+'-test.npz')
    else:
        d1 = np.load(loc+s+'-train-proc.npz')
        d2 = np.load(loc+s+'-test-proc.npz')

    d = {'train': {'x': th.from_numpy(d1['data']), 'y': th.from_numpy(d1['labels'])},
        'val': {'x': th.from_numpy(d2['data']), 'y': th.from_numpy(d2['labels'])}}
    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        cv.RandomHorizontalFlip(),
        cv.Pad(4, 2),
        cv.RandomCrop(sz),
        # cutout,
        lambda x: x.transpose(2,0,1),
        th.from_numpy
        ])

    return d, augment

def cifar10(opt):
    return cifar_helper(opt, 'cifar10')

def cifar100(opt):
    return cifar_helper(opt, 'cifar100')

def imagenet32(opt):
    return cifar_helper(opt, 'imagenet32')

def svhn(opt):
    loc = home + '/local2/pratikac/svhn/'

    d1 = sio.loadmat(loc + 'train_32x32.mat')
    d2 = sio.loadmat(loc + 'extra_32x32.mat')
    d3 = sio.loadmat(loc + 'test_32x32.mat')

    d = {'train': { 'x': np.concatenate([d1['X'],d2['X']], axis=3).astype(np.float32),
                    'y': np.concatenate([d1['y'],d2['y']])-1},
        'val': {'x': d3['X'].astype(np.float32),
                'y': d3['y']-1}}

    mean = np.array([109.9, 109.7, 113.8])[None,:,None,None]
    std = np.array([50.1, 50.6, 50.9])[None,:,None,None]

    for k in d:
        d[k]['x'] = np.transpose(d[k]['x'], (3,2,0,1))
        d[k]['x'] = (d[k]['x'] - mean)/std

        d[k]['x'] = th.from_numpy(d[k]['x'])
        d[k]['y'] = th.from_numpy(d[k]['y'])

    shuffle_data(d['train'])

    sz = d['train']['x'].size(3)
    augment = tnt.transform.compose([
        lambda x: x.numpy().astype(np.float32),
        lambda x: x.transpose(1,2,0),
        cv.RandomHorizontalFlip(),
        cv.Pad(4, 2),
        cv.RandomCrop(sz),
        cv.CutOut(14, (0,0,0)),
        lambda x: x.transpose(2,0,1),
        th.from_numpy
        ])

    return d, lambda x: x

def imagenet(opt, only_train=False):
    loc = home + '/local2/pratikac/imagenet'
    bsz, nw = opt['b'], 4

    traindir = os.path.join(loc, 'train')
    valdir = os.path.join(loc, 'val')

    input_transform = [transforms.Scale(256)]

    normalize = [T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])]

    train_folder = datasets.ImageFolder(traindir, transforms.Compose([
            T.RandomSizedCrop(224),
            T.RandomHorizontalFlip()] + normalize))
    train_loader = th.utils.data.DataLoader(
        train_folder,
        batch_size=bsz, shuffle=True,
        num_workers=nw, pin_memory=True)

    val_folder = datasets.ImageFolder(valdir, transforms.Compose(
            input_transform + [transforms.CenterCrop(224)] + normalize))
    val_loader = th.utils.data.DataLoader(
        val_folder,
        batch_size=bsz, shuffle=False,
        num_workers=nw, pin_memory=True)

    ids = th.arange(0, len(train_loader)).long()

    return [dict(train=train_loader,
                val=val_loader,
                test=val_loader,
                train_full=train_loader,
                idx=ids) for i in range(opt['n'])]
