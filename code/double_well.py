from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
sns.set_color_codes()

plt.ion()

fsz = 18
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)

ls = [100]

for i in range(len(ls)):
    l = ls[i]

    plt.figure(1, figsize=(7,7))
    plt.clf()

    d = 1.5

    # Contour Plot
    y,x = np.mgrid[-d:d:20j, -d:d:20j]

    # Vector Field
    phi = (x**2-1)**2/4. + y**2/2.
    u = -x**3 + x + l*np.exp(phi - (x**2 + y**2)**2/4.)*(-y)
    v = -y + l*np.exp(phi - (x**2 + y**2)**2/4.)*(x)

    s = np.sqrt(u**2 + v**2) + 1e-6
    # u /= s
    # v /= s

    plt.streamplot(x,y,u,v, density=0.5, color='k', linewidth=5*s / s.max())
    plt.grid()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    plt.axes().set_aspect('equal')
    # plt.savefig('../fig/double_well%d.pdf'%(i+1), bbox_inches='tight')