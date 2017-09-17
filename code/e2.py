import hickle as hkl
import torch as th

import sys

f = sys.argv[1]

d = th.load(f)
print d['opt']

d['S'] = d['S'].numpy()
d['fgrad'] = d['fgrad'].numpy()

hkl.dump(d, f+'.hkl')