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

import io_utils
import duck_3D
import forward_sim
import phasing_3d
import maps

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def phase(mapper, iters_str = '100DM 100ERA'):
    """
    phase a crappy crystal diffraction volume
    
    Parameters
    ----------
    mapper : object
        A class object that can be used by 3D-phasing, which
        requires the following methods:
            I     = mapper.Imap(modes)   # mapping the modes to the intensity
            modes = mapper.Pmod(modes)   # applying the data projection to the modes
            modes = mapper.Psup(modes)   # applying the support projection to the modes
            O     = mapper.object(modes) # the main object of interest
            dict  = mapper.finish(modes) # add any additional output to the info dict
    
    Keyword Arguments
    -----------------
    iters_str : str, optional, default ('100DM 100ERA')
        supported iteration strings, in general it is '[number][alg][space]'
        [N]DM [N]ERA 1cheshire
    """
    alg_iters = config_iters_to_alg_num(iters_str)
    
    eMod = []
    eCon = []
    O = mapper.O
    for alg, iters in alg_iters :
        
        print(alg, iters)
        
        if alg == 'ERA':
           O, info = phasing_3d.ERA(iters, mapper = mapper)
         
        if alg == 'DM':
           O, info = phasing_3d.DM(iters, mapper = mapper)
        
        if alg == 'cheshire':
           O, info = mapper.scans_cheshire(O, steps=[1,1,1])
         
        eMod += info['eMod']
        eCon += info['eCon']
    
    return O, mapper, eMod, eCon, info

def parse_cmdline_args(default_config='phase.ini'):
    parser = argparse.ArgumentParser(description="phase a crappy crystal from it's diffraction intensity. The results are output into a .h5 file.")
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
    ################
    if params['input_file'] is None :
        f = h5py.File(args.filename)

    I = f[params['data']][()]
    
    if params['solid_unit'] is None :
        solid_unit = None
    else :
        print('loading solid_unit from file...')
        solid_unit = f[params['solid_unit']][()]
    
    if params['mask'] is None :
        mask = None
    else :
        mask = f[params['mask']][()]
    
    if type(params['voxels']) != int and params['voxels'][0] == '/'  :
        voxels = f[params['voxels']][()]
    else :
        voxels = params['voxels']

    # make the mapper
    #################
    mapper = maps.Mapper_ellipse(f[params['data']][()], 
                                 Bragg_weighting   = f[params['bragg_weighting']][()], 
                                 diffuse_weighting = f[params['diffuse_weighting']][()], 
                                 solid_unit        = solid_unit,
                                 voxels            = voxels,
                                 overlap           = params['overlap'],
                                 unit_cell         = params['unit_cell'],
                                 space_group       = params['space_group'],
                                 alpha             = params['alpha'],
                                 dtype             = params['dtype']
                                 )
    f.close()

    # phase
    #######
    O, mapper, eMod, eCon, info = phase(mapper, params['iters'])


    # output
    ########
    outputdir = os.path.split(os.path.abspath(args.filename))[0]

    # mkdir if it does not exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    f = h5py.File(args.filename)
    
    group = '/phase'
    if group not in f:
        f.create_group(group)
    
    # solid unit
    key = group+'/solid_unit'
    if key in f :
        del f[key]
    f[key] = O
    
    # real-space crystal
    key = group+'/crystal'
    if key in f :
        del f[key]
    f[key] = mapper.sym_ops.solid_to_crystal_real(O)

    del info['eMod']
    del info['eCon']
    info['eMod'] = eMod
    info['eCon'] = eCon
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
