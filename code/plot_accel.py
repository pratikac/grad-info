import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, glob2, pdb, argparse
import cPickle as pickle
import seaborn as sns
import torch as th
import pandas as pd
import matplotlib.cm as cm

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


plt.figure(1, figsize=(7,7))
plt.clf()

d = 2
l = 10

# Contour Plot
x1,y1 = np.linspace(-d,d,20), np.linspace(-d,d,20)
y,x = np.mgrid[-d:d:20j, -d:d:20j]

# Vector Field
alpha, beta = 0.25, 2
phi = -x**2/2. - 2*alpha*x*y - beta*y**2/2.

u = -x - 2*alpha*y
v = -2*alpha*x - beta*y
s = np.sqrt(u**2 + v**2) + 1e-6
u /= s
v /= s

u1 = -x - 2*alpha*y + l*np.exp(-phi - (x**2 + y**2)**2/4.)*(-2*alpha*x - beta*y)
v1 = -2*alpha*x - beta*y + l*np.exp(-phi - (x**2 + y**2)**2/4.)*(x + 2*alpha*y)
s1 = np.sqrt(u1**2 + v1**2) + 1e-6
u1 /= s1
v1 /= s1

ax = plt.gcf()
plt.streamplot(x,y,u,v, density=0.15, color='k', linewidth=7*s/s.max())
plt.streamplot(x1,y1,u1,v1, start_points=[[-1,-1]], color='r', linewidth=7*s1/s1.max())
plt.grid()
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xticks([])
plt.yticks([])
# plt.xticks([-2,0,2])
# plt.yticks([-2,0,2])
plt.axes().set_aspect('equal')

plt.contour(x, y, phi, levels=np.linspace(phi.min(), phi.max(), 6), cmap=cm.Blues, alpha=0.4)
plt.savefig('../fig/j_accel.pdf', bbox_inches='tight')