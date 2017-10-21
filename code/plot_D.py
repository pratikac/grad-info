import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, glob2, pdb, argparse
import cPickle as pickle
import seaborn as sns
import torch as th

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

def lenets():
    loc = 'lenets/(Oct_19_16_14_22)_opt_{"b":128,"d":0.1,"dataset":"mnist","m":"lenets","s":42}'
    d = loaddir(os.path.join(opt['l'], loc), force=opt['f'])

    plt.figure(1, figsize=(8,8))
    plt.clf()

    bins = np.logspace(-10, -1, 10)
    idxs = [1, 2, 5]
    for ii,di in enumerate(d):
        if ii not in idxs:
            continue

        lw, ec = 2, 'w'
        sns.distplot(d[ii]['eig'], bins=bins,
            kde=False, color='k',
            hist_kws=dict(alpha=alphas[ii], edgecolor=ec, linewidth=lw)
            )

        if alphas[ii] < 0.5:
            lw, ec = 1, 'k'
            sns.distplot(d[ii]['eig'], bins=bins,
                kde=False, color='k',
                hist_kws=dict(histtype='step', alpha=1, edgecolor=ec, linewidth=lw)
                )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([1e-10, 0.1])
    plt.ylim([0, 1e5])

    plt.xticks(np.logspace(-10, -1, 4))
    plt.yticks([1, 1e2, 1e4])

    plt.grid()

    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    if opt['s']:
        plt.savefig('../fig/lenets_D.pdf', bbox_inches='tight')


def cifar10():

    loc = 'allcnns/(Sep_08_02_44_57)_opt_{"b":128,"dataset":"cifar10","m":"allcnns","s":42}'
    # loc = 'allcnns/(Oct_20_21_47_33)_opt_{"b":128,"d":0.0,"dataset":"cifar10","m":"allcnns","s":42}'
    d = loaddir(os.path.join(opt['l'], loc), force=opt['f'])

    plt.figure(2, figsize=(8,8))
    plt.clf()

    bins = np.logspace(-15, 2, 15)
    idxs = [1,2,5]
    for ii,di in enumerate(d):
        if ii not in idxs:
            continue

        lw, ec = 2, 'w'
        sns.distplot(d[ii]['eig'], bins=bins,
            kde=False, color='k',
            hist_kws=dict(alpha=alphas[ii], edgecolor=ec, linewidth=lw)
            )

        if alphas[ii] < 0.5:
            lw, ec = 1, 'k'
            sns.distplot(d[ii]['eig'], bins=bins,
                kde=False, color='k',
                hist_kws=dict(histtype='step', alpha=1, edgecolor=ec, linewidth=lw)
                )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([1e-12, 1])
    plt.ylim([0, 1e5])

    plt.xticks(np.logspace(-12, 0, 4))
    plt.yticks([1, 1e2, 1e4])

    plt.grid()

    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    if opt['s']:
        plt.savefig('../fig/allcnns_cifar10_D.pdf', bbox_inches='tight')


def cifar100():
    loc = 'allcnns/(Sep_08_02_45_19)_opt_{"b":128,"dataset":"cifar100","m":"allcnns","s":42}'
    d = loaddir(os.path.join(opt['l'], loc), force=opt['f'])

    plt.figure(3, figsize=(8,8))
    plt.clf()

    bins = np.logspace(-15, 2, 15)
    idxs = [1, 2, 5]
    for ii,di in enumerate(d):
        if ii not in idxs:
            continue

        lw, ec = 2, 'w'
        sns.distplot(d[ii]['eig'], bins=bins,
            kde=False, color='k',
            hist_kws=dict(alpha=alphas[ii], edgecolor=ec, linewidth=lw)
            )

        if alphas[ii] < 0.5:
            lw, ec = 1, 'k'
            sns.distplot(d[ii]['eig'], bins=bins,
                kde=False, color='k',
                hist_kws=dict(histtype='step', alpha=1, edgecolor=ec, linewidth=lw)
                )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim([1e-12, 1])
    plt.ylim([0, 1e5])

    plt.xticks(np.logspace(-12, 0, 4))
    plt.yticks([1, 1e2, 1e4])

    plt.grid()

    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    if opt['s']:
        plt.savefig('../fig/allcnns_cifar100_D.pdf', bbox_inches='tight')


lenets()
cifar10()
cifar100()