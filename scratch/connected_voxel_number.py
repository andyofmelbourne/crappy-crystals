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
import os, sys

# load a test image

f = h5py.File('../hdf5/pdb/pdb.h5')
a = f['forward_model_pdb/solid_unit'][()]
support0 = f['forward_model_pdb/support'][()]
b = f['phase/solid_unit'][()]

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

from maps import choose_N_highest_pixels

# update support
################
# find sigma such that:
# s_blur = s .conv. e^{-r^2 / 2 sigma**2}
# 0 < sum_r[ s_blur > median(s_blur)] - N < tol
# then apply the voxel number support within this region

def find_sigma_thresh(array, N=32445, tol=10, sigma=2., maxIters=100):
    from scipy.ndimage.filters import gaussian_filter
    
    support    = np.zeros(array.shape, dtype=np.bool)
    array_blur = gaussian_filter(array, sigma, mode='wrap')
    array_blur_max = array_blur.max()
    
    s1 = 0.
    s0 = array_blur_max 
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        
        threshold = s * array_blur_max
        
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

def 

def get_inside_edge_indices(array, structure=None, iterations=1, mask=None, \
                            output=None, border_value=0, origin=0, brute_force=False):

def weighted_binary_dilation(input, weights=None, structure=None, iterations=1, mask=None, \
                             output=None, border_value=0, origin=0, brute_force=False):
    """

    """

N = 32445
sup, sigma = find_sigma_thresh(np.abs(a)**2, N=N*1.5, tol = 100)


sup2 = choose_N_highest_pixels(np.abs(a), N, support = sup)




