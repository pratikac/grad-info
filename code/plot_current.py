import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import seaborn as sns
import torch as th
from future.utils import lmap

sns.set_style('ticks')
sns.set_color_codes()

plt.ion()

p = argparse.ArgumentParser('')
p.add_argument('-i', type=str, default='', help='location')
p.add_argument('-f', help='force', action='store_true')
opt = vars(p.parse_args())

fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)

def set_ticks(xt=[], xts=[], yt=[], yts=[]):
    if len(xt):
        if not len(xts):
            xts = [str(s) for s in xt]
        plt.xticks(xt, xts)
    if len(yt):
        if not len(yts):
            yts = [str(s) for s in yt]
        plt.yticks(yt, yts)

def cuda_fft(x):
    import pytorch_fft.fft as fft

    xr, ximg = fft.rfft(th.from_numpy(x).cuda())
    p = (xr**2 + ximg**2)**0.5
    freq = np.fft.rfftfreq(x.shape[1])
    return p.cpu().numpy(), freq


def xcorr(x):
    """FFT based autocorrelation function, which is faster than numpy.correlate"""
    # x is supposed to be an array of sequences, of shape (totalelements, length)

    fftx = np.fft.fft(x, axis=1)
    ret = np.fft.ifft(fftx*np.conjugate(fftx), axis=1).real
    ret = np.fft.fftshift(ret, axes=1)
    return ret

def autocorrelation(x):
    n = len(x)
    mean = np.mean(x)
    c0 = np.sum((x - mean) ** 2) / float(n)

    def r(h):
        return ((x[:n - h] - mean) *
                (x[h:] - mean)).sum() / float(n) / c0

    lag = np.arange(0,n,n//100)
    y = lmap(r, lag)
    return lag, y

if opt['f']:
    d = th.load(opt['i'])

    print 'Will rewrite: ', opt['i']

    d['ddw'] = d['w'][:,1:] - d['w'][:,:-1]
    d['mom_avg'] = (d['mom'][:,1:] + d['mom'][:,:-1])/2.
    d['cth'] = (d['ddw']*d['mom_avg'])/(np.linalg.norm(d['ddw'], axis=0)*np.linalg.norm(d['mom_avg'], axis=0))
    d['ddw_par'] = d['ddw']*d['cth']
    d['sth'] = (1-d['cth']**2)**0.5
    d['ddw_perp'] = d['ddw']*d['sth']

    print 'Start autocorrelation:'
    d['lag'], _ = autocorrelation(d['w'][0])
    d['ac'] = []
    for i in xrange(d['w'].shape[0]):
        _, ac = autocorrelation(d['w'][i])
        d['ac'].append(ac)
        if i % 10:
            print i
    d['ac'] = np.array(d['ac'])

    # keys = ['lag', 'ac', 'w', 'dw','freq']
    # th.save( {k:v for k,v in d.items() if k in d and k in keys}, opt['i']+'_ac.pz')
    # sys.exit(0)

    # if not 'freq' in d:
    print 'Start FFT'
    d['p'], d['freq'] = cuda_fft(d['w'])
    d['pdw'], _ = cuda_fft(d['ddw'])
    d['pdw_par'], _ = cuda_fft(d['ddw_par'])
    d['pdw_perp'], _ = cuda_fft(d['ddw_perp'])

    print 'Saving: ', opt['i']
    print 'Continue?'
    raw_input()
    th.save(d, opt['i'])
    sys.exit(0)

def plot_fft():
    d = th.load('/Users/pratik/Dropbox/iclr18data/traj/(Oct_12_01_41_39)_opt_{"s":42}_trajectory_extra.pz')

    plt.figure(1, figsize=(8,7))
    plt.clf()

    idx = range(1000)
    sns.tsplot(time=d['freq'][idx], data=d['pdw'][:,idx], color='k')
    plt.xscale('log')

    plt.xlim([1e-5, 1e-2])
    plt.ylim([0, 0.15])
    plt.yticks([0, 0.05, 0.1, 0.15])

    # plt.title('FFT of dx(t)')
    plt.xlabel('frequency (1/epoch)')
    plt.ylabel('amplitude')

    plt.grid()
    plt.savefig('../fig/fft_fcnet.pdf', bbox_inches='tight')

def plot_ac():
    d = th.load('/Users/pratik/Dropbox/iclr18data/traj/(Oct_12_01_41_39)_opt_{"s":42}_trajectory.pz_ac.pz')

    plt.figure(2, figsize=(8,7))
    plt.clf()

    d['ac'] = d['ac'][~np.isnan(d['ac']).any(axis=1)]
    sns.tsplot(time=d['lag'], data=d['ac'], color='k')

    plt.xscale('log')
    plt.xlim([1e3, 1e5])
    plt.ylim([-0.2, 1])
    plt.yticks([-0.2, 0.2, 0.6, 1.0])

    # plt.title('Autocorrelation of x(t)')
    plt.xlabel('lag (epochs)')
    plt.ylabel('auto-correlation')

    n = d['w'].shape[1]
    ax = plt.gca()
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='r')
    ax.axhline(y=z95 / np.sqrt(n), linestyle='--', color='r')
    # ax.axhline(y=0.0, lw=2, linestyle='-', color='r')
    ax.axhline(y=-z95 / np.sqrt(n), linestyle='--', color='r')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='r')

    plt.grid()
    plt.savefig('../fig/ac_fcnet.pdf', bbox_inches='tight')


def plot_grad():
    d = th.load('/Users/pratik/Dropbox/iclr18data/traj/(Oct_12_01_41_39)_opt_{"s":42}_trajectory_extra.pz')

    g = d['dw']
    norm_g = np.linalg.norm(g, axis=0)/np.sqrt(g.shape[0])
    norm_gi = np.abs(g)

    n = g.shape[1]
    idx = np.arange(0, n, n//1000)
    norm_g, norm_gi = norm_g[idx], norm_gi[:,idx]

    ng = pd.ewma(pd.DataFrame(norm_g), com=10).as_matrix().flatten()
    ngi = pd.ewma(pd.DataFrame(norm_gi), com=1000).as_matrix()

    plt.figure(3, figsize=(8,7))
    plt.clf()

    sns.tsplot(time=idx, data=ng, color='k') #, condition=r'g', legend=True)
    # sns.tsplot(time=idx, data=ngi, color='k', condition=r'g^i', legend=True)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e2, 1e5])
    plt.ylim([1e-4, 1e-2])

    plt.xlabel('epochs')
    plt.ylabel(r'|grad f| / sqrt(d)')

    plt.grid()
    plt.savefig('../fig/fgrad_fcnet.pdf', bbox_inches='tight',rasterized=True)

plot_fft()
plot_ac()
plot_grad()