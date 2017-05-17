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

f = h5py.File('../hdf5/duck/duck.h5')
a = f['phase/solid_unit'][()]
support0 = f['forward_model/support'][()]

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
    
    return support

def get_outside_edge_indices(binary_mask, **kwargs):
    #from scipy.ndimage.morphology import binary_dilation
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(len(binary_mask.shape), 1)
    
    # dilate the binary mask 
    #b2 = binary_dilation(binary_mask, structure=struct)
    b2 = ndimage.morphology.grey_dilation(binary_mask, footprint=struct, mode='wrap').astype(np.bool)

    # get the indices of the difference
    i = np.where(np.bitwise_xor(binary_mask, b2).ravel())[0]
    return i

def get_inside_edge_indices(binary_mask, **kwargs):
    #from scipy.ndimage.morphology import binary_erosion
    #from scipy.ndimage.morphology import grey_erosion
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(len(binary_mask.shape), 1)
    
    # dilate the binary mask 
    #b2 = binary_erosion(binary_mask, structure=struct)
    b2 = ndimage.morphology.grey_erosion(binary_mask, footprint=struct, mode='wrap').astype(np.bool)

    # get the indices of the difference
    i = np.where(np.bitwise_xor(binary_mask, b2).ravel())[0]
    return i

def center_rho_roll(rho, shift = None):
    """Move electron density map so its center of mass aligns near the center of the grid by rolling array
    Author: Thomas D. Grant"""
    if shift is None :
        rhocom = np.array(ndimage.measurements.center_of_mass(rho))
        gridcenter = np.array(rho.shape)/2.
        shift = gridcenter-rhocom
        shift = shift.astype(int)
    
    rho = np.roll(np.roll(np.roll(rho, shift[0], axis=0), shift[1], axis=1), shift[2], axis=2)
    return rho, shift

def downsample_array(array, N):
    out = array.copy()
    for i in range(len(array.shape)):
        # transpose 'out' so that the shrunk dimension is last
        out = np.swapaxes(out, i, -1)
        # reshape and sum the last axis
        newshape = tuple(list(out.shape[:-1]) + [out.shape[-1]//N, N])
        out      = np.sum(out.reshape( newshape ), axis=-1)
        # transpose back to original dims
        out = np.swapaxes(out, i, -1)
    return out

def upsample_array(array, N):
    out = array.copy()
    for i in range(len(array.shape)):
        # transpose 'out' so that the shrunk dimension is first
        out = np.swapaxes(out, i, 0)
        out = np.array([out for i in range(N)]).T
        newshape      = list(out.shape[:-1])
        newshape[-1] *= N
        out           = out.reshape( newshape ).T 
        # transpose back to original dims
        out = np.swapaxes(out, i, 0)
    return out



def voxel_number_support_single_connected_region(intensity, N, init_sup=None, downsample=None, i_max=1000):
    sups = []
    if downsample is not None :
        intensity2 = downsample_array(intensity, downsample)
        sup, sups0 = voxel_number_support_single_connected_region(intensity2, int(N/float(downsample**len(intensity.shape))))
        init_sup   = upsample_array(sup, downsample)
        sups.append(init_sup.copy())
        

    # sigma threshold
    from scipy import spatial, ndimage 
    struct = ndimage.generate_binary_structure(len(intensity.shape), 1)
    
    if init_sup is None :
        sup = find_sigma_thresh(intensity, N=N, sigma=1.0, tol = 100)
        
        # label the regions and select the one with the largest intensity
        labeled, num_labels = ndimage.measurements.label(sup)
        
        intensities = [np.sum(intensity[labeled==i]) for i in range(1, num_labels+1)]
        sup.fill(False) 
        sup[labeled==(np.argmax(intensities)+1)] = True
    else :
        sup = init_sup.copy()
    
    for i in range(i_max):
        changed = False
        
        # if we have too many voxels then discard inner shell voxels
        N_sup = np.sum(sup)
        if N_sup > N :
            # raveled indices
            inside         = get_inside_edge_indices(sup)
            inside_sorted  = inside[np.argsort( intensity.ravel()[inside] )]
             
            #print('removing', N_sup-N, 'inner shell voxels from the support')
            if (N_sup - N) >= len(inside_sorted):
                sup.ravel()[inside_sorted] = False
            else :
                sup.ravel()[inside_sorted[N_sup - N]] = False
            changed = True
            sups.append(sup)
        
        # if we have too few voxels then add to the outer shell
        elif N_sup < N :
            # raveled indices
            outside        = get_outside_edge_indices(sup)
            outside_sorted = outside[np.argsort( intensity.ravel()[outside] )]
            
            #print('adding', N-N_sup, 'outer shell voxels to the support')
            if (N - N_sup) >= len(outside_sorted):
                sup.ravel()[outside_sorted] = True
            else :
                sup.ravel()[outside_sorted[N - N_sup]] = True
            changed = True
            sups.append(sup.copy())

        # if we have exactly the right number of voxels
        # then replace inner shell voxels with outer shell voxels
        else :
            # raveled indices
            inside         = get_inside_edge_indices(sup)
            inside_sorted  = inside[np.argsort( intensity.ravel()[inside] )]
            temp_sup       = sup.copy()
            
            # loop over coords from smallest to largest
            # get the most intense outside voxel 
            # and swap the supports 
            for j in inside_sorted :
                # mask the weakest the voxel
                temp_sup.ravel()[j] = False
                
                # now get the outer shell 
                outside = get_outside_edge_indices(temp_sup)
                
                # find the index of the most intense value
                k = outside[np.argmax(intensity.ravel()[outside])]
                
                # unmask that voxel if it is greater
                if intensity.ravel()[k] > intensity.ravel()[j]:
                    #print('swapping indices:', j, '<-->', k, 'in support')
                    changed = True
                    temp_sup.ravel()[k] = True
                    sups.append(temp_sup.copy())
                else :
                    # we are done with this inner shell
                    #print('done with inner shell:', i)
                    temp_sup.ravel()[j] = True
                    sups.append(temp_sup.copy())
                    break
            
            sup = temp_sup.copy()
        
        if changed is False :
            break
    return sup, sups

N = np.sum(support0)
#support = voxel_number_support_single_connected_region(np.abs(a)**2, N, downsample=2)

"""
sups = []
# simple blur -> threshold
# should use larger sigma
sup = find_sigma_thresh(intensity, N=int(N*0.1), sigma=1.0, tol = 100)
sups.append(sup.copy())

# label the regions and select the one with the largest intensity
labeled, num_labels = ndimage.measurements.label(sup)

intensities = [np.sum(intensity[labeled==i]) for i in range(1, num_labels+1)]
sup.fill(False) 
sup[labeled==(np.argmax(intensities)+1)] = True
sups.append(sup.copy())


# centre the array
#_        , shift = center_rho_roll(sup*intensity)
#intensity, shift = center_rho_roll(intensity, shift)
#sup      , shift = center_rho_roll(sup, shift)


# take the least intense inside vox's 
# and put them on the most intense outside edges
for i in range(10000):
    # re-center the intensity and support
    # back to zero
    #intensity, shift = center_rho_roll(intensity, -shift)
    #sup, shift       = center_rho_roll(sup, shift)
    # shift
    #_, shift         = center_rho_roll(intensity*sup)
    #intensity, shift = center_rho_roll(intensity, shift)
    #sup, shift       = center_rho_roll(sup, shift)
    changed = False

    # if we have too many voxels then discard inner shell voxels
    N_sup = np.sum(sup)
    if N_sup > N :
        # raveled indices
        inside         = get_inside_edge_indices(sup)
        inside_sorted  = inside[np.argsort( intensity.ravel()[inside] )]
         
        print('removing', N_sup-N, 'inner shell voxels from the support')
        if (N_sup - N) >= len(inside_sorted):
            sup.ravel()[inside_sorted] = False
        else :
            sup.ravel()[inside_sorted[N_sup - N]] = False
        changed = True
        sups.append(sup.copy())
    
    # if we have too few voxels then add to the outer shell
    elif N_sup < N :
        # raveled indices
        outside        = get_outside_edge_indices(sup)
        outside_sorted = outside[np.argsort( intensity.ravel()[outside] )]
        
        print('adding', N-N_sup, 'outer shell voxels to the support')
        if (N - N_sup) >= len(outside_sorted):
            sup.ravel()[outside_sorted] = True
        else :
            sup.ravel()[outside_sorted[N - N_sup]] = True
        changed = True
        sups.append(sup.copy())

    # if we have exactly the right number of voxels
    # then replace inner shell voxels with outer shell voxels
    else :
        # raveled indices
        inside         = get_inside_edge_indices(sup)
        inside_sorted  = inside[np.argsort( intensity.ravel()[inside] )]
        temp_sup       = sup.copy()
        
        # loop over coords from smallest to largest
        # get the most intense outside voxel 
        # and swap the supports 
        for j in inside_sorted :
            # mask the weakest the voxel
            temp_sup.ravel()[j] = False
            
            # now get the outer shell 
            outside = get_outside_edge_indices(temp_sup)
            
            # find the index of the most intense value
            k = outside[np.argmax(intensity.ravel()[outside])]
            
            # unmask that voxel if it is greater
            if intensity.ravel()[k] > intensity.ravel()[j]:
                print('swapping indices:', j, '<-->', k, 'in support')
                changed = True
                temp_sup.ravel()[k] = True
            else :
                # we are done with this inner shell
                print('done with inner shell:', i)
                temp_sup.ravel()[j] = True
                break
            sups.append(temp_sup.copy())
        
        sup = temp_sup.copy()
    
    if changed is False :
        break
"""

#sup, shift       = center_rho_roll(sup, -shift)
#sups.append(sup.copy())
"""
i = get_inside_edge_indices(support)
inside    = np.zeros_like(support)
inside[np.unravel_index(i, support.shape)] = True
"""

#i = get_outside_edge_indices(support)
#outside    = np.zeros_like(support)
#outside[i] = True

#N = 32445
#sup, sigma = find_sigma_thresh(np.abs(a)**2, N=N*1.5, tol = 100)
#sup2 = choose_N_highest_pixels(np.abs(a), N, support = sup)




