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

d = th.load(opt['i'])
w = d['w']

d['ddw'] = d['w'][:,1:] - d['w'][:,:-1]
d['mom'] = (d['mom'][:,1:] + d['mom'][:,:-1])/2.
d['cth'] = (d['ddw']*d['mom'])/(np.linalg.norm(d['ddw'], axis=0)*np.linalg.norm(d['mom'], axis=0))
d['ddw_par'] = d['ddw']*d['cth']
d['sth'] = (1-d['cth']**2)**0.5
d['ddw_perp'] = d['ddw']*d['sth']

# if not 'freq' in d:
print 'Start FFT'
d['p'] = np.abs(np.fft.rfftn(w))
d['pdw'] = np.abs(np.fft.rfftn(d['ddw']))
d['freq'] = np.fft.rfftfreq(w.shape[1])

print 'Saving back to: ', opt['i']
th.save(d, opt['i'])

# plt.figure(1)
# plt.clf()

# # plt.loglog(d['freq'][:1000], d['p'][100][:1000],'k')

# sns.tsplot(time=d['freq'][:1000], data=d['p'][:,:1000], ci=[68,95])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([1e-5,1e-2])

# plt.grid()
