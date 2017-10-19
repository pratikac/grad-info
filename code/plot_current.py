import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch as th

import seaborn as sns
sns.set_style('ticks')
sns.set_color_codes()

plt.ion()

p = argparse.ArgumentParser('')
p.add_argument('-i', type=str, default='', help='location')
opt = vars(p.parse_args())

assert not opt['i'] == ''

d = th.load(opt['i'])
w = d['w']

# print 'Start FFT'
# d['p'] = np.abs(np.fft.rfftn(w))
# d['freq'] = np.fft.rfftfreq(w.shape[1])

# print 'Saving back to: ', opt['i']
# th.save(d, opt['i'])

plt.figure(1)
plt.clf()
plt.grid()

plt.plot(d['freq'][:1000], d['p'][10][:1000],'k')
plt.xscale('log')
plt.yscale('log')
