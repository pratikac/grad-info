import torch as th
import torchvision as thv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, logging, pdb
from copy import deepcopy
import exptutils
import numpy as np
from torch.nn.parallel import scatter, parallel_apply, gather

from microbn import MicroBatchNorm2d, MicroBatchNorm1d

def get_num_classes(opt):
    d = dict(mnist=10, svhn=10, cifar10=10,
            cifar100=100, imagenet=1000, imagenet32=1000, halfmnist=10)
    if not opt['dataset'] in d:
        assert False, 'Unknown dataset: %s'%opt['dataset']
    return d[opt['dataset']]

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    name = 'mnistfc'
    def __init__(self, opt):
        super(mnistfc, self).__init__()

        c = 1024
        opt['d'] = 0.2
        opt['l2'] = 0.

        self.m = nn.Sequential(
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        if opt['v']:
            print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class lenet(nn.Module):
    name = 'lenet'
    def __init__(self, opt, c1=20, c2=50, c3=500, microbn=False):
        super(lenet, self).__init__()

        opt['d'] = opt.get('d', -1.0)
        if opt['d'] < 0:
            opt['d'] = 0.25
        opt['l2'] = 0.

        if microbn:
            bn1, bn2 = MicroBatchNorm1d, MicroBatchNorm2d
        else:
            bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                bn2(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,c1,5,3,opt['d']),
            convbn(c1,c2,5,2,opt['d']),
            View(c2*2*2),
            nn.Linear(c2*2*2, c3),
            bn1(c3),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c3,10))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class fclenet(nn.Module):
    name = 'fclenet'
    def __init__(self, opt, c=64):
        super(fclenet, self).__init__()

        opt['l2'] = 0.
        nc = opt.get('nc', get_num_classes(opt))

        self.m = nn.Sequential(
            View(49),
            nn.Linear(49,c),
            nn.BatchNorm1d(c),
            nn.ReLU(True),
            nn.Linear(c, nc)
        )

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class fclenett(fclenet):
    name = 'fclenett'
    def __init__(self, opt, c=16):
        opt['d'] = 0.0
        opt['l2'] = 0
        super(fclenett, self).__init__(opt, c)

class fclenets(fclenet):
    name = 'fclenets'
    def __init__(self, opt, c=64):
        opt['d'] = 0.0
        opt['l2'] = 0
        super(fclenets, self).__init__(opt, c)

class lenett(nn.Module):
    name = 'lenett'
    def __init__(self, opt, c1=4, c2=8):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0

        super(lenett, self).__init__()

        opt['l2'] = 0.
        nc = opt.get('nc', 10)

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,c1,5,3,opt['d']),
            convbn(c1,c2,5,2,opt['d']),
            View(c2*2*2),
            nn.Linear(c2*2*2, nc))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class lenets(lenet):
    name = 'lenets'
    def __init__(self, opt, c1=8, c2=16, c3=128):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0

        super(lenets, self).__init__(opt, c1, c2, c3)

class lenetl(lenet):
    name = 'lenetl'
    def __init__(self, opt, c1=40, c2=100, c3=1000):
        opt['d'] = 0.5
        super(lenetl, self).__init__(opt, c1, c2, c3)

class cifarcnns(nn.Module):
    name = 'cifarcnns'

    def __init__(self, opt):
        super(cifarcnns, self).__init__()

        if (not 'l2' in opt) or opt['l2'] < 0:
            opt['l2'] = 1e-3

        if (not 'd' in opt) or opt['d'] < 0:
            if opt['augment']:
                opt['d'] = 0.0
            else:
                opt['d'] = 0.25

        num_classes = get_num_classes(opt)
        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                bn2(co),
                nn.ReLU(True))

        self.m = nn.Sequential(
            convbn(3,16,5,1,2),
            nn.MaxPool2d(2,2),
            convbn(16,20,5,1,2),
            nn.MaxPool2d(2,2),
            convbn(20,20,5,1,2),
            nn.MaxPool2d(2,2),
            View(320),
            nn.Linear(320, num_classes))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class allcnn(nn.Module):
    name = 'allcnn'

    def __init__(self, opt, c1=96, c2=192, microbn=False):
        super(allcnn, self).__init__()

        if (not 'd' in opt) or opt['d'] < 0:
            if opt['augment']:
                opt['d'] = 0.0
            else:
                opt['d'] = 0.5

        if (not 'l2' in opt) or opt['l2'] < 0:
            opt['l2'] = 1e-3

        num_classes = get_num_classes(opt)

        if microbn:
            bn1, bn2 = MicroBatchNorm1d, MicroBatchNorm2d
        else:
            bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                bn2(co),
                nn.ReLU(True)
                )
        self.m = nn.Sequential(
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,num_classes,1,1),
            nn.AvgPool2d(8),
            View(num_classes))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class allcnntt(allcnn):
    name = 'allcnntt'
    def __init__(self, opt, c1=4, c2=8):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0

        opt['l2'] = 1e-4
        super(allcnntt, self).__init__(opt, c1, c2)

class allcnnt(allcnn):
    name = 'allcnnt'
    def __init__(self, opt, c1=8, c2=16):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0

        opt['l2'] = 1e-4
        super(allcnnt, self).__init__(opt, c1, c2)

class allcnns(allcnn):
    name = 'allcnns'
    def __init__(self, opt, c1=12, c2=24):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0

        opt['l2'] = 1e-4
        super(allcnns, self).__init__(opt, c1, c2)

class allcnnl(allcnn):
    name = 'allcnnl'
    def __init__(self, opt, c1=120, c2=240):
        super(allcnnl, self).__init__(opt, c1, c2)

class caddtable_t(nn.Module):
    def __init__(self, m1, m2):
        super(caddtable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return th.add(self.m1(x), self.m2(x))

class wideresnet(nn.Module):
    name = 'wideresnet'
    @staticmethod
    def block(ci, co, s, p=0.):
        h = nn.Sequential(
                nn.BatchNorm2d(ci),
                nn.ReLU(inplace=True),
                nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Dropout(p),
                nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False))
        if ci == co:
            return caddtable_t(h, nn.Sequential())
        else:
            return caddtable_t(h,
                        nn.Conv2d(ci, co, kernel_size=1, stride=s, padding=0, bias=False))

    @staticmethod
    def netblock(nl, ci, co, blk, s, p=0.):
        ls = [blk((i==0 and ci or co), co, (i==0 and s or 1), p) for i in range(nl)]
        return nn.Sequential(*ls)

    def __init__(self, opt):
        super(wideresnet, self).__init__()

        if opt['d'] < 0:
            opt['d'] = 0.25
        if opt['l2'] < 0:
            opt['l2'] = 5e-4

        d, depth, widen = opt['d'], opt['depth'], opt['widen']

        num_classes = get_num_classes(opt)

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)//6

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False),
                self.netblock(n, nc[0], nc[1], self.block, 1, d),
                self.netblock(n, nc[1], nc[2], self.block, 2, d),
                self.netblock(n, nc[2], nc[3], self.block, 2, d),
                nn.BatchNorm2d(nc[3]),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8),
                View(nc[3]),
                nn.Linear(nc[3], num_classes))

        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, math.sqrt(2./m.in_features))
                m.bias.data.zero_()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class wrn101(wideresnet):
    name ='wrn101'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 10,1
        super(wrn101, self).__init__(opt)

class wrn521(wideresnet):
    name ='wrn521'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 52,1
        super(wrn521, self).__init__(opt)

class wrn164(wideresnet):
    name ='wrn164'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 16,4
        super(wrn164, self).__init__(opt)

class wrn168(wideresnet):
    name ='wrn168'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 16,8
        super(wrn168, self).__init__(opt)

class wrn2810(wideresnet):
    name ='wrn2810'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 28, 10
        super(wrn2810, self).__init__(opt)

class resnet18(nn.Module):
    name = 'resnet18'
    def __init__(self, opt):
        super(resnet18, self).__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet18(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet50(nn.Module):
    name = 'resnet50'
    def __init__(self, opt):
        super(resnet50, self).__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet50(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet101(nn.Module):
    name = 'resnet101'
    def __init__(self, opt):
        super(resnet101, self).__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet101(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet152(nn.Module):
    name = 'resnet152'
    def __init__(self, opt):
        super(resnet152, self).__init__(num_classes=get_num_classes(opt))
        opt['l2'] = 1e-4
        self.m = thv.models.resnet152()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class alexnet(nn.Module):
    name = 'alexnet'
    def __init__(self, opt):
        super(alexnet, self).__init__()
        self.m = getattr(thv.models, opt['m'])()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet121(nn.Module):
    name = 'densenet121'
    def __init__(self, opt):
        super(densenet121, self).__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet121(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet169(nn.Module):
    name = 'densenet169'
    def __init__(self, opt):
        super(densenet169, self).__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet169(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet201(nn.Module):
    name = 'densenet201'
    def __init__(self, opt):
        super(densenet201, self).__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet201(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class squeezenet(nn.Module):
    name = 'squeezenet'
    def __init__(self, opt):
        super(squeezenet, self).__init__()

        self.m = getattr(thv.models, 'squeezenet1_1')(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class cattable_t(nn.Module):
    def __init__(self, m1, m2):
        super(cattable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return th.cat((self.m1(x), self.m2(x)), 1)

class densenet(nn.Module):
    name = 'densenet'

    @staticmethod
    def bottleneck(nc, gr, p):
        ic = 4*gr
        h = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, ic, kernel_size=1, bias=False),
            nn.BatchNorm2d(ic),
            nn.ReLU(True),
            nn.Conv2d(ic, gr, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p),
            )
        return cattable_t(h, nn.Sequential())

    @staticmethod
    def basic(nc, gr, p):
        h = nn.Sequential(
            nn.BatchNorm2d(nc),
            nn.ReLU(True),
            nn.Conv2d(nc, gr, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p)
            )
        return cattable_t(h, nn.Sequential())

    @staticmethod
    def transition(nci, nco, p):
        return nn.Sequential(
            nn.BatchNorm2d(nci),
            nn.ReLU(True),
            nn.Conv2d(nci, nco, kernel_size=1, bias=False),
            nn.Dropout(p),
            nn.AvgPool2d(2))

    def __init__(self, opt):
        super(densenet, self).__init__()

        gr = opt.get('gr', 12)
        depth = opt.get('depth', 100)
        reduction = opt.get('reduction', 0.5)
        is_bottleneck = True

        assert not opt['dataset'] == 'imagenet', 'Use specific densenet from torchvision, \
                this is only meant for cifar'
        num_classes = get_num_classes(opt)

        if opt['d'] < 0:
            opt['d'] = 0.0
        if opt['l2'] < 0:
            opt['l2'] = 1e-4

        nblk = (depth-4) // 3
        if is_bottleneck:
            nblk //= 2

        ncis, ncos = [2*gr], [2*gr]
        for i in range(1,4):
            nc = ncos[-1] + nblk*gr
            ncis.append(nc)
            ncos.append(int(math.floor(nc*reduction)))

        def denseblock(nc, gr, nblk, blk, p):
            wl = self.bottleneck if is_bottleneck else self.basic
            ls = [wl(nc + i*gr, gr, p) for i in range(nblk)]
            return nn.Sequential(*ls)

        self.m = nn.Sequential(
                nn.Conv2d(3, ncis[0], kernel_size=3, padding=1, bias=False),
                denseblock(ncis[0], gr, nblk, is_bottleneck, opt['d']),
                self.transition(ncis[1], ncos[1], opt['d']),
                denseblock(ncos[1], gr, nblk, is_bottleneck, opt['d']),
                self.transition(ncis[2], ncos[2], opt['d']),
                denseblock(ncos[2], gr, nblk, is_bottleneck, opt['d']),
                nn.BatchNorm2d(ncis[3]),
                nn.ReLU(True),
                nn.AvgPool2d(8),
                View(ncis[3]),
                nn.Linear(ncis[3], num_classes)
            )

        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)
