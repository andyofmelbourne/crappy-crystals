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

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import pyximport; pyximport.install()
from ellipse_2D_cython import project_2D_Ellipse_arrays_cython
from ellipse_2D_cython_new import project_2D_Ellipse_arrays_cython_test

import io_utils
import duck_3D
import forward_sim
import phasing_3d
import maps

def parse_cmdline_args(default_config='ellipse_projections.ini'):
    parser = argparse.ArgumentParser(description="Perform ellipse projections on x, y in Wx x**2 + Wy * y**2 = I space. The results are output into a .h5 file.")
    parser.add_argument('-f', '--filename', type=str, \
                        help="file name of the output *.h5 file to edit / create")
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
    
    # check that the output file was specified
    ################################################
    if args.filename is None and params['output_file'] is not None :
        fnam = params['output_file']
        args.filename = fnam
    
    if args.filename is None :
        raise ValueError('output_file in the ini file is not valid, or the filename was not specified on the command line')
    
    return args, params



if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    # make the input
    Wx = np.array([params['wx']])
    Wy = np.array([params['wy']])
    I  = np.array([params['i']])
    x = np.array([params['x']])
    y = np.array([params['y']])

    if params['random_xy'] is not None and params['random_xy'] is not False :
        if Wx > 0 and Wy > 0 and I > 0 :
            r = max(np.sqrt(I)/np.sqrt(Wx), np.sqrt(I)/np.sqrt(Wy))
        else :
            r = 1.

        rand_x  = np.random.random(params['random_xy'])*2.*r - r
        rand_y  = np.random.random(params['random_xy'])*2.*r - r
        Wx_rand = np.empty_like(rand_x)
        Wy_rand = np.empty_like(rand_x)
        I_rand  = np.empty_like(rand_x)
        mask    = np.empty_like(rand_x).astype(np.uint8)
        Wx_rand[:] = Wx
        Wy_rand[:] = Wy
        I_rand[:]  = I
        mask[:]    = 1
        xp_rand, yp_rand = project_2D_Ellipse_arrays_cython_test(rand_x, rand_y, Wx_rand, Wy_rand, I_rand, mask)
    else :
        xp_rand, yp_rand = None, None 

    
    # project
    #########
    xp, yp = project_2D_Ellipse_arrays_cython_test(x, y, Wx, Wy, I, np.array([1], dtype=np.uint8))

    # output
    ########
    outputdir = os.path.split(os.path.abspath(args.filename))[0]

    # mkdir if it does not exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    print('writing to:', args.filename)
    f = h5py.File(args.filename)
    
    group = '/ellipse_projections'
    if group not in f:
        f.create_group(group)

    if xp_rand is not None :
        # x_rand
        key = group+'/x_rand'
        if key in f :
            del f[key]
        f[key] = rand_x
        
        # y_rand
        key = group+'/y_rand'
        if key in f :
            del f[key]
        f[key] = rand_y
        
        # xp_rand
        key = group+'/xp_rand'
        if key in f :
            del f[key]
        f[key] = xp_rand
        
        # yp
        key = group+'/yp_rand'
        if key in f :
            del f[key]
        f[key] = yp_rand
    else :
        # x_rand
        key = group+'/x_rand'
        if key in f :
            del f[key]
        
        # y_rand
        key = group+'/y_rand'
        if key in f :
            del f[key]
        
        # xp_rand
        key = group+'/xp_rand'
        if key in f :
            del f[key]
        
        # yp
        key = group+'/yp_rand'
        if key in f :
            del f[key]
    
    # I
    key = group+'/I'
    if key in f :
        del f[key]
    f[key] = I
    
    # Wx
    key = group+'/Wx'
    if key in f :
        del f[key]
    f[key] = Wx

    # Wy
    key = group+'/Wy'
    if key in f :
        del f[key]
    f[key] = Wy
    
    # yp
    key = group+'/yp'
    if key in f :
        del f[key]
    f[key] = yp

    # x
    key = group+'/x'
    if key in f :
        del f[key]
    f[key] = x
    
    # y
    key = group+'/y'
    if key in f :
        del f[key]
    f[key] = y

    # xp
    key = group+'/xp'
    if key in f :
        del f[key]
    f[key] = xp
    
    # yp
    key = group+'/yp'
    if key in f :
        del f[key]
    f[key] = yp

    f.close() 
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print(e)
