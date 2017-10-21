import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, pdb, argparse
import cPickle as pickle
import seaborn as sns
import torch as th

sns.set_style('ticks')
sns.set_color_codes()

plt.ion()

p = argparse.ArgumentParser('')
p.add_argument('-i', type=str, default='', help='location')
p.add_argument('-f', help='force', action='store_true')
opt = vars(p.parse_args())

fsz = 24
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

assert not opt['i'] == ''


def cuda_fft(x):
    import pytorch_fft.fft as fft

    xr, ximg = fft.rfft(th.from_numpy(x).cuda())
    p = (xr**2 + ximg**2)**0.5
    freq = np.fft.rfftfreq(x.shape[1])
    return p.cpu().numpy(), freq

d = th.load(opt['i'])

if opt['f']:
    print 'Force: '%opt['f']
    print 'Will rewrite: ', opt['i']

    d['ddw'] = d['w'][:,1:] - d['w'][:,:-1]
    d['mom_avg'] = (d['mom'][:,1:] + d['mom'][:,:-1])/2.
    d['cth'] = (d['ddw']*d['mom_avg'])/(np.linalg.norm(d['ddw'], axis=0)*np.linalg.norm(d['mom_avg'], axis=0))
    d['ddw_par'] = d['ddw']*d['cth']
    d['sth'] = (1-d['cth']**2)**0.5
    d['ddw_perp'] = d['ddw']*d['sth']

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

plt.figure(1, figsize=(8,7))
plt.clf()

idx = range(500)
sns.tsplot(time=d['freq'][idx], data=d['pdw'][:,idx], color='k')
plt.xscale('log')

plt.xlim([1e-5, 2e-2])
plt.ylim([0, 0.15])
plt.yticks([0, 0.05, 0.1, 0.15])

plt.title('FFT of dx(t)')
plt.xlabel('frequency (1/epoch)')
plt.ylabel('amplitude')

plt.grid()
# plt.savefig('../fig/fft_fcnet.pdf', bbox_inches='tight')