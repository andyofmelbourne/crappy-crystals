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
from calculate_constraint_ratio import calculate_constraint_ratio

def choose_N_highest_pixels(array, N, tol = 1.0e-10, maxIters=1000, syms = None, support = None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)

    If support is not None then values outside the support
    are ignored. 
    """
    
    # no overlap constraint
    if syms is not None :
        # if array is not the maximum value
        # of the M symmetry related units 
        # then do not update 
        max_support = syms[0] > ((1-tol) * np.max(syms[1:], axis=0))
    else :
        max_support = np.ones(array.shape, dtype = np.bool)
    
    if support is not None and support is not 1 :
        sup = support
        a = array[(max_support > 0) * (support > 0)]
    else :
        sup = True
        a = array[(max_support > 0)]
    
    # search for the cutoff value
    s0 = array.max()
    s1 = array.min()
    
    failed = False
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = np.sum(a > s) - N
          
        if e == 0 :
            #print('e==0, exiting...')
            break
        
        if e < 0 :
            s0 = s
        else :
            s1 = s

        #print(s, s0, s1, e)
        if np.abs(s0 - s1) < tol and np.abs(e) > 0 :
            failed = True
            print('s0==s1, exiting...', s0, s1, np.abs(s0 - s1), tol)
            break
        
    S = (array > s) * max_support * sup
    
    # if failed is True then there are a lot of 
    # entries in a that equal s
    #if failed :
    if False :
        print('failed, sum(max_support), sum(S), voxels, pixels>0:',np.sum(max_support), np.sum(S), N, np.sum(array>0), len(a>0))
        # if S is less than the 
        # number of voxels then include 
        # some of the pixels where array == s
        count      = np.sum(S)
        ii, jj, kk = np.where((np.abs(array-s)<=tol) * (max_support * sup > 0))
        l          = N - count
        print(count, N, l, len(ii))
        if l > 0 :
            S[ii[:l], jj[:l], kk[:l]]    = True
        else :
            S[ii[:-l], jj[:-l], kk[:-l]] = False
    
    #print('number of pixels in support:', np.sum(S), i, s, e)
    return S

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
        
    # if params['solid_unit'] is a path to a h5 file then use that
    elif params['solid_unit'][-3:] == '.h5':
        f = h5py.File(params['solid_unit'], 'r')
        solid_unit = f['solid_unit'][()].astype(np.float64)
        f.close()

        # Hack! mask it 
        solid_unit *= solid_unit.real > 0.05
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

    # remove overlap
    #############################################
    # make crystal
    O = solid_unit
    intensity = (O * O.conj()).real
    
    U = np.fft.ifftn(info['modes'], axes=(1,2,3))
    U = (U * U.conj()).real
    
    # make the crystal
    Crystal = []
    for i in [0, info['sym'].unitcell_size[0]]:
        for j in [0, info['sym'].unitcell_size[1]]:
            for k in [0, info['sym'].unitcell_size[2]]:
                for ii in range(U.shape[0]):
                    Crystal.append(np.roll(U[ii], (i,j,k), (0,1,2)))
    
    syms = np.array(Crystal)
    
    #Crystal = np.zeros_like(info['unit_cell'])
    #for i in [0, unit_cell[0]]:
    #    for j in [0, unit_cell[1]]:
    #        for k in [0, unit_cell[2]]:
    #            Crystal += np.roll(info['unit_cell'], (i,j,k), (0,1,2))
    #syms = np.array([solid_unit, Crystal])
    S = choose_N_highest_pixels(intensity, info['voxels'], syms = syms)
    
    solid_unit *= S
    
    print('\nRemoving overlap of:', info['voxels'] - np.sum(S),'voxels')
    
    # calculate the diffraction data and metadata
    #############################################
    diff, info = forward_sim.generate_diff(solid_unit, unit_cell, N, sigma, **params)
    
    # make crystal
    Crystal = np.zeros_like(info['unit_cell'])
    for i in [0, unit_cell[0]]:
        for j in [0, unit_cell[1]]:
            for k in [0, unit_cell[2]]:
                Crystal += np.roll(info['unit_cell'], (i,j,k), (0,1,2))
    
    # calculate the constraint ratio
    ################################
    omega_con, omega_Bragg, omega_global  = calculate_constraint_ratio(info['support'], params['space_group'], unit_cell)
    info['omega_continuous'] = omega_con
    info['omega_Bragg']      = omega_Bragg
    info['omega_global']     = omega_global
                               
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


    # for testing
    dxyz = np.array([0.575625, 0.63875, 1.0959375])
    Uxyz = np.array([36.840, 40.880, 70.140])
    geom = {'vox': dxyz, 'abc': Uxyz, 'originx': np.array([0,0,0])}
    fnam = 'test_U.ccp4'
    import write_ccp4
    write_ccp4.write_ccp4(np.fft.fftshift(info['unit_cell'].real.astype(np.float32)), fnam, geom, SGN=19)

    fnam = 'test_C.ccp4'
    write_ccp4.write_ccp4(np.fft.fftshift(Crystal.real.astype(np.float32)), fnam, geom, SGN=19)
