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

try :
    import ConfigParser as configparser 
except ImportError :
    import configparser 

import numpy as np
import h5py 
import argparse
import os, sys
import re
import copy

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path = [os.path.join(root, 'utils')] + sys.path

import io_utils
import duck_3D
import forward_sim
import phasing_3d
# testing
import maps_gpu as maps
#import maps
import fidelity


if __name__ == '__main__':

    f = h5py.File('../hdf5/pdb/pdb.h5', 'r')
    # data
    I = f[params['data']][()]
    
    # solid unit
    if params['solid_unit'] is None :
        solid_unit = None
    else :
        print('loading solid_unit from file...')
        solid_unit = f[params['solid_unit']][()]
    
    # detector mask
    if params['mask'] is None :
        mask = None
    else :
        mask = f[params['mask']][()]
    
    # voxel support
    if params['voxels'] is None :
        voxels = None
    elif type(params['voxels']) != int and params['voxels'][0] == '/'  :
        voxels = f[params['voxels']][()]
    else :
        voxels = params['voxels']
    
    # voxel_sup_blur support
    if params['voxel_sup_blur'] is None :
        voxel_sup_blur = None
    elif type(params['voxel_sup_blur']) != float and params['voxel_sup_blur'][0] == '/'  :
        voxel_sup_blur = f[params['voxel_sup_blur']][()]
    else :
        voxel_sup_blur = params['voxel_sup_blur']
    
    # voxel_sup_blur_frac support
    if params['voxel_sup_blur_frac'] is None :
        voxel_sup_blur_frac = None
    elif type(params['voxel_sup_blur_frac']) != float and params['voxel_sup_blur_frac'][0] == '/'  :
        voxel_sup_blur_frac = f[params['voxel_sup_blur_frac']][()]
    else :
        voxel_sup_blur_frac = params['voxel_sup_blur_frac']
    
    # support update frequency
    if params['support_update_freq'] is None :
        support_update_freq = None
    elif type(params['support_update_freq']) != int and params['support_update_freq'][0] == '/'  :
        support_update_freq = f[params['support_update_freq']][()]
    else :
        support_update_freq = params['support_update_freq']

    # fixed support
    if params['support'] is None or params['support'] is False :
        support = None
    else :
        support = f[params['support']][()]
        
    # Bragg weighting
    if params['bragg_weighting'] is None or params['bragg_weighting'] is False :
        bragg_weighting = None
    else :
        bragg_weighting = f[params['bragg_weighting']][()]

    # Diffuse weighting
    if params['diffuse_weighting'] is None or params['diffuse_weighting'] is False :
        diffuse_weighting = None
    else :
        diffuse_weighting = f[params['diffuse_weighting']][()]

    # Unit cell parameters
    if type(params['unit_cell']) != int and params['unit_cell'][0] == '/'  :
        unit_cell = f[params['unit_cell']][()]
    else :
        unit_cell = params['unit_cell']
    
    
    # make the mapper
    #################
    #solid_unit = support * np.random.random(f[params['data']].shape) + 0J
    I0 = f[params['data']][()]
    mapper = maps.Mapper_ellipse(f[params['data']][()], 
                                 Bragg_weighting   = bragg_weighting, 
                                 diffuse_weighting = diffuse_weighting, 
                                 solid_unit        = solid_unit,
                                 voxels            = voxels,
                                 voxel_sup_blur    = voxel_sup_blur,
                                 voxel_sup_blur_frac = voxel_sup_blur_frac,
                                 overlap           = params['overlap'],
                                 support           = support,
                                 support_update_freq = support_update_freq,
                                 unit_cell         = unit_cell,
                                 space_group       = params['space_group'],
                                 alpha             = params['alpha'],
                                 dtype             = params['dtype']
                                 )
