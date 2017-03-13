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

import h5py
import numpy as np
import scipy.ndimage

import os, sys
# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import io_utils
import forward_sim
import duck_3D

# load the example

# make the symmetry operator

# look at the unit-cell mapping and its inverse


config = configparser.ConfigParser()
config.read('../hdf5/duck/forward_model.ini')

params = io_utils.parse_parameters(config)['forward_model']

# make the solid unit
#####################
if params['solid_unit'] == 'duck':
    duck       = duck_3D.make_3D_duck(shape = params['shape'])

    # flip the solid unit axes
    if 'flip' in params.keys():
        flip = params['flip']
        duck = duck[::flip[0], ::flip[1], ::flip[2]].copy()
    
    # transpose the solid unit in the unit cell
    if 'transpose' in params.keys():
        duck = np.transpose(duck, params['transpose']).copy()
    
    solid_unit = np.zeros(params['detector'], dtype=np.complex)
    solid_unit[:duck.shape[0], :duck.shape[1], :duck.shape[2]] = duck
    
    # position the solid unit in the unit cell
    solid_unit = np.roll(solid_unit, params['position'][0], 0)
    solid_unit = np.roll(solid_unit, params['position'][1], 1)
    solid_unit = np.roll(solid_unit, params['position'][2], 2)
    
else :
    raise ValueError("solid_unit not supported, can only be 'duck' at this point...")

# make the input
################
unit_cell  = params['unit_cell']
N          = params['n']
sigma      = params['sigma']
del params['unit_cell']
del params['n']
del params['sigma']
del params['solid_unit']

# calculate the diffraction data and metadata
#############################################
diff, info = forward_sim.generate_diff(solid_unit, unit_cell, N, sigma, **params)

sym = info['sym']

syms = sym.solid_syms_real(solid_unit)

for s in syms :
    U0 = sym.solid_syms_Fourier(np.fft.fftn(s))
    I0  = info['diffuse_weighting'] * np.sum(np.abs(U0)**2, axis=0)
    I0 += info['Bragg_weighting']   * np.abs(np.sum(U0, axis=0))**2
    print(np.allclose(I0, diff))
