# python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import os
try :
    import ConfigParser as configparser
except :
    import configparser

import time
import re

import crappy_crystals
from crappy_crystals import utils
from crappy_crystals.phasing.era import ERA
from crappy_crystals.phasing.dm import DM

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters


def phase(I, solid_support, params, good_pix = None, solid_known = None):

    # Type of sample support
    if 'support' in params['phasing'].keys() and params['phasing']['support'] == 'voxel_number':
        support = params['voxel_number']['n']
        print('sample update: voxel number with', support, 'voxels')
    else :
        support = solid_support
        print('sample update: fixed support with ', np.sum(support), 'voxels')
    
    # background
    background = None
    if 'background' in params['phasing'].keys() :
        if params['phasing']['background'] is True :
            background = True

        print('background:', background, params['phasing']['background'])
    
    d0 = time.time()
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
    
    solid_ret = None
    eMod = []
    for alg, iters in alg_iters :

        print(alg, iters)

        if alg == 'ERA':
            solid_ret, info = ERA(I, iters, support, params, \
                                  mask = good_pix, O = solid_ret, \
                                  background = background, method = 1, hardware = params['phasing']['hardware'], \
                                  alpha = 1.0e-10, dtype = 'double', full_output = True)
                    
            eMod += info['eMod']
            if 'background' in info.keys():
                background = info['background']
            else :
                info['r_av'] = None
        
        if alg == 'DM':
            solid_ret, info = DM(I, iters, support, params, \
                                  mask = good_pix, O = solid_ret, \
                                  background = background, method = 1, hardware = params['phasing']['hardware'], \
                                  alpha = 1.0e-10, dtype = 'double', full_output = True)

            eMod += info['eMod']
            if 'background' in info.keys():
                background = info['background']
            else :
                info['r_av'] = None

    d1 = time.time()
    print('\n\nTime (s):', d1 - d0)
    
    return solid_ret, info['I'], info['support'], info['r_av'], np.array(eMod), np.zeros_like(eMod)


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    kwargs = utils.io_utils.read_input_output_h5(args.input)
    
    solid_ret, diff_ret, support_ret, B_rav, emod, efid = phase(kwargs['data'], kwargs['sample_support'], kwargs['config_file'], \
                                good_pix = kwargs['good_pix'], solid_known = kwargs['solid_unit'])
    
    # write the h5 file 
    fnam = os.path.join(kwargs['config_file']['output']['path'], 'output.h5')
    utils.io_utils.write_input_output_h5(fnam, data = kwargs['data'], \
            data_retrieved = diff_ret, sample_support = kwargs['sample_support'], \
            sample_support_retrieved = support_ret, good_pix = kwargs['good_pix'], \
            solid_unit = kwargs['solid_unit'], solid_unit_retrieved = solid_ret, modulus_error = emod, \
            fidelity_error = efid, config_file = kwargs['config_file_name'], B_rav = B_rav)
