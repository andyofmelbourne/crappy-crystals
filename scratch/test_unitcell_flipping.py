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

# look at the real-space tiling of the unit-cell
Ur = np.sum(sym.solid_syms_real(solid_unit), axis=0)

# look at the Fourier-space tiling of the unit-cell (in real-space)
Ufs = sym.solid_syms_Fourier(np.fft.fftn(solid_unit))

d  = np.sum(sym.unflip_modes_Fourier(Ufs, apply_translation=True), axis=0)
d  = np.fft.ifftn(d)

Uf = np.sum(Ufs, axis=0)
Uf = np.fft.ifftn(Uf)

import pyqtgraph as pg
pg.show(np.fft.fftshift(np.sum(d, axis=1)).real[:,::-1])
pg.show(np.fft.fftshift(np.sum(Uf, axis=1)).real[:,::-1])
pg.show(np.fft.fftshift(np.sum(Ur, axis=1)).real[:,::-1])
input('Enter')
