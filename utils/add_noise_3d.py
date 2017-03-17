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

import numpy as np

def R_scale_data(diff, is_fft_shifted = True):
    if is_fft_shifted is False :
        diff_out = np.fft.ifftshift(diff.copy())
    else :
        diff_out = diff.copy()
    
    i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
    j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
    k = np.fft.fftfreq(diff.shape[2]) * diff.shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    R       = np.sqrt(i**2 + j**2 + k**2)
    R[0, 0, 0] = 0.5
    
    # R scaling
    R_scale   = 3. * ((R+1.)**2 - R**2) / ((R+1.)**3 - R**3)
    diff_out  = diff_out * R_scale
    return diff_out, R_scale

def mask_courners(shape, is_fft_shifted=True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    R       = np.sqrt(i**2 + j**2 + k**2)
    
    # mask courners 
    mask = np.ones(shape, dtype = np.bool)
    if remove_courners :
        l           = np.where(R >= np.min(shape) / 2.)
        mask[l]     = False
    return mask

def add_poisson_noise(diff, n, renormalise=True):
    # normalise
    norm     = np.sum(diff)
    diff_out = diff.copy() / norm
    
    # Poisson sampling
    diff_out = np.random.poisson(lam = float(n) * diff_out).astype(np.float64)

    if renormalise :
        diff_out = diff_out / np.sum(diff_out) * norm
    return diff_out
    
    

def add_noise_3d(diff, n, is_fft_shifted = True, remove_courners = True, unit_cell_size=None):
    """
    Add Poisson noise to a 3d volume.

    This function takes into account the
    reduced counting statistics at higher 
    resolutions.

    n = is the mean number of photons detected 
        in a speckle at the middle edge of the 
        detector. Assuming oversampling of 2
        in each direction.
    
    Expected number of photons for voxel R:
    = I(R) 3u [(R+u)^2 - R^2] / [(R+u)^3 - R^3]
    
    for a square pixel of side length u. This 
    depends somewhat on the merging strategy.
    To first order in u this becomes:
    = I(R) 2u / R
    
    This is not valid for the courners of the 
    detector that are sampled much less.
    """
    # R-scale
    diff_out, R_scale = R_scale_data(diff, is_fft_shifted)

    # poisson sampling
    diff_out = add_poisson_noise(diff_out, n, renormalise=False)
    
    # un-R-scale
    diff_out /= R_scale
    
    # renormalise
    diff_out = diff_out / np.sum(diff_out) * np.sum(diff)
    
    return diff_out

def rad_av(diff, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(diff.shape[0]) * diff.shape[0]
        j = np.fft.fftfreq(diff.shape[1]) * diff.shape[1]
        k = np.fft.fftfreq(diff.shape[2]) * diff.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16).ravel()
        
        if is_fft_shifted is False :
            rs = np.fft.ifftshift(rs)
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, diff.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    return r_av
