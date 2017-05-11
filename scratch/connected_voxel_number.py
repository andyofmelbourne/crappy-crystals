#!/usr/bin/env python

# for python 2 / 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try :
    range = xrange
except NameError :
    pass

import h5py
import numpy as np

# load a test image

f = h5py.File('../hdf5/pdb/pdb.h5')
a = f['forward_model_pdb/solid_unit'][()]
support0 = f['forward_model_pdb/support'][()]
b = f['phase/solid_unit'][()]


# update support
################
# find sigma such that:
# s_blur = s .conv. e^{-r^2 / 2 sigma**2}
# 0 < sum_r[ s_blur > median(s_blur)] - N < tol
# then apply the voxel number support within this region


def find_sigma_thresh(array, N=32445, tol=10, thresh=0.1, maxIters=100):
    from scipy.ndimage.filters import gaussian_filter
    
    support    = np.zeros(array.shape, dtype=np.bool)
    array_blur = np.empty_like(array)
    
    s0 = 0.
    s1 = np.array(array.shape).max() / 8
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        
        gaussian_filter(array, s, output=array_blur, mode='wrap', truncate=2.0)
        threshold = thresh * array_blur.max()
        
        support = array_blur > threshold
        e = np.sum(support) - N
          
        if np.abs(e) <= tol :
            #print('e==0, exiting...')
            break
        
        if e < 0 :
            s0 = s
        else :
            s1 = s

    return support, s







