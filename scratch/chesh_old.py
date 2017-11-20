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
import h5py

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import io_utils
import duck_3D
import forward_sim
import maps

if __name__ == '__main__':
    # process config file
    config = configparser.ConfigParser()
    config.read('chesh.ini')
    
    params = io_utils.parse_parameters(config)['forward_model']
    
    # make the solid unit
    #####################
    if params['solid_unit'] == 'duck':
        duck       = duck_3D.make_3D_duck(shape = params['shape'])
        solid_unit = np.zeros(params['detector'], dtype=np.complex)
        solid_unit[:duck.shape[0], :duck.shape[1], :duck.shape[2]] = duck

        # transpose the solid unit in the unit cell
        if 'transpose' in params.keys():
            solid_unit = np.transpose(solid_unit, params['transpose'])
        
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
                               
    # make the mapper
    #################
    mapper = maps.Mapper_ellipse(diff, 
                                 Bragg_weighting = info['Bragg_weighting'], 
                                 diffuse_weighting = info['diffuse_weighting'], 
                                 solid_unit = solid_unit,
                                 voxels = info['voxels'], 
                                 sym = info['sym'], 
                                 overlap = 'unit_cell',
                                 )
    
    # do the cheshire scan
    ######################
    #solid_chesh, info_chesh = mapper.scans_cheshire(solid_unit, steps=[32,1,1], unit_cell=True)
    solid_chesh, info_chesh = mapper.scans_cheshire(solid_unit, steps=[1,1,32], unit_cell=True)

    # save the error map to file
    f = h5py.File('cheshire_scan.h5')
    if 'error_map' in f :
        del f['error_map']
    if 'crystal' in f :
        del f['crystal']
        
    f['error_map'] = info_chesh['error_map']
    f['crystal'] = info['crystal']
    f.close()


