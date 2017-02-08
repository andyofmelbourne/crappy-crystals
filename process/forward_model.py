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

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import io_utils
import duck_3D
import forward_sim

def parse_cmdline_args(default_config='forward_model.ini'):
    parser = argparse.ArgumentParser(description='calculate the forward model diffraction intensity for a disorded crystal. The results are output into a .h5 file.')
    parser.add_argument('-f', '--filename', type=str, \
                        help="file name of the *.h5 file to edit / create")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # if config is non then read the default from the *.h5 dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], default_config)
        if not os.path.exists(args.config):
            args.config = '../process/' + default_config
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = io_utils.parse_parameters(config)[default_config[:-4]]
    
    return args, params


if __name__ == '__main__':
    args, params = parse_cmdline_args()

    # check that the output file was specified
    ##########################################
    if args.filename is not None :
        fnam = args.filename
    elif params['output_file'] is not None :
        fnam = params['output_file']
    else :
        raise ValueError('output_file in the ini file is not valid, or the filename was not specified on the command line')
    
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
                               
    # output
    ########
    outputdir = os.path.split(os.path.abspath(args.filename))[0]

    # mkdir if it does not exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    f = h5py.File(fnam)
    
    group = '/forward_model'
    if group not in f:
        f.create_group(group)
    
    # diffraction data
    key = group+'/data'
    if key in f :
        del f[key]
    f[key] = diff
    
    # solid unit
    key = group+'/solid_unit'
    if key in f :
        del f[key]
    f[key] = solid_unit
    
    # everything else
    for key, value in info.items():
        if value is None :
            continue 
        
        h5_key = group+'/'+key
        if h5_key in f :
            del f[h5_key]
        
        try :
            print('writing:', h5_key, type(value))
            f[h5_key] = value
        
        except Exception as e :
            print('could not write:', h5_key, ':', e)
        
    f.close() 
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print(e)
