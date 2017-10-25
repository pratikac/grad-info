import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, glob2, pdb, argparse
import cPickle as pickle
import seaborn as sns
import torch as th
import pandas as pd

sns.set_style('ticks')
sns.set_color_codes()

parser = argparse.ArgumentParser(description='Plot D spectrum')
parser.add_argument('-l',
            help='location', type=str,
            default='/Users/pratik/Dropbox/iclr18data')
parser.add_argument('-f',
            help='reprocess data',
            action='store_true')
parser.add_argument('-s',
            help='save figures',
            action='store_true')
opt = vars(parser.parse_args())

fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)

alphas = np.logspace(-1, -0.2, 6)

bins = np.logspace(-5, 2, 10)
xlim, ylim = [1e-5, 1e2], [0.1, 1e5]
xticks, yticks = [1e-5, 1e-3, 1e-1, 1e1], [1, 1e2, 1e4]

# bins = np.logspace(-2, 2, 10)
# xlim, ylim = [1e-2, 1e2], [0.1, 1e5]
# xticks, yticks = [1e-2, 1, 1e2], [1, 1e2, 1e4]


def set_lim():
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.grid()

    plt.xlabel('eigenvalues')
    plt.ylabel('frequency')

def set_ticks(xt=[], xts=[], yt=[], yts=[]):
    if len(xt):
        if not len(xts):
            xts = [str(s) for s in xt]
        plt.xticks(xt, xts)
    if len(yt):
        if not len(yts):
            yts = [str(s) for s in yt]
        plt.yticks(yt, yts)

def loaddir(dir, expr='/*', force=False):
    pkl = dir+'/all.pz'

    if (not force) and os.path.isfile(pkl):
        return pickle.load(open(pkl, 'r'))

    print 'Pattern: ', dir + expr + '.eig.pz'
    fs = sorted(glob2.glob(dir + expr + '.eig.pz'))
    d = []

    for f in fs:
        di = th.load(f)
        d.append(di)
        print f

    pickle.dump(d, open(pkl, 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    return d

def rough(idx, loc, fname):
    d = loaddir(os.path.join(opt['l'], loc), force=opt['f'])

    plt.figure(idx, figsize=(8,8))
    plt.clf()


    idxs = [1, 2, 5]
    for ii,di in enumerate(d):
        if ii not in idxs:
            continue

        l = d[ii]['eig']
        thresh = l.max() * len(l) * np.finfo(np.float32).eps
        e = pd.DataFrame(d[ii]['eig']).clip(lower=thresh)
        print ''
        print fname
        print 'rank: ', d[ii]['rank']
        print 'thresh: ', thresh
        print e.describe()
        continue

        lw, ec = 2, 'w'
        sns.distplot(np.clip(d[ii]['eig'], a_min=thresh, a_max=None), bins=bins,
            kde=False, color='k',
            hist_kws=dict(alpha=alphas[ii], edgecolor=ec, linewidth=lw)
            )

        if alphas[ii] < 0.5:
            lw, ec = 1, 'k'
            sns.distplot(np.clip(d[ii]['eig'], a_min=thresh, a_max=None), bins=bins,
                kde=False, color='k',
                hist_kws=dict(histtype='step', alpha=1, edgecolor=ec, linewidth=lw)
                )

    set_lim()

    if opt['s']:
        plt.savefig('../fig/%s.pdf'%fname, bbox_inches='tight')

def lenets():
    rough(  idx=1,
            loc='lenets/(Oct_19_16_14_22)_opt_{"b":128,"d":0.1,"dataset":"mnist","m":"lenets","s":42}',
            fname='lenets_D')

def fclenets():
    rough(  idx=2,
            loc='fclenets/(Oct_22_14_22_50)_opt_{"b":128,"d":0.0,"dataset":"halfmnist","m":"fclenets","s":42}',
            fname='smallfc_D')

def cifar10():
    rough(  idx=3,
            loc='allcnns/(Sep_08_02_44_57)_opt_{"b":128,"dataset":"cifar10","m":"allcnns","s":42}',
            fname='allcnns_cifar10_D')

def cifar10_augment():
    rough(  idx=4,
            loc='allcnns/(Sep_08_02_44_57)_opt_{"b":128,"dataset":"cifar10","m":"allcnns","s":42}/augment',
            fname='allcnns_cifar10_augment_D')

def convs():
    rough(  idx=5,
            loc='cifarcnns/(Oct_22_04_23_08)_opt_{"b":128,"d":0.25,"dataset":"cifar10","m":"cifarcnns","s":42}',
            fname='convs_D')

def cifar100():
    rough(  idx=6,
            loc='allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}',
            fname='allcnns_cifar100_D')

# lenets()
# fclenets()
cifar10()
# cifar100()

# cifar10_augment()
# convs()