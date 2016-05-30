import numpy as np
import sys
import os
import ConfigParser
import time

import crappy_crystals
import crappy_crystals.utils
from crappy_crystals import utils
from crappy_crystals import phasing
from crappy_crystals.phasing.maps import *
from crappy_crystals.phasing.era import ERA


def phase(I, solid_support, params, good_pix = None, solid_known = None):
    
    d0 = time.time()
    solid_ret, info = ERA(I, params['phasing']['era'], solid_support, params, \
                          mask = good_pix, O = None, \
                          background = None, method = 1, hardware = 'cpu', \
                          alpha = 1.0e-10, dtype = 'double', full_output = True)
    d1 = time.time()
    print '\n\nTime (s):', d1 - d0
    
    return solid_ret, info['I'], np.array(info['eMod']), np.zeros_like(info['eMod'])

if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    kwargs = utils.io_utils.read_input_output_h5(args.input)
    
    solid_ret, diff_ret, emod, efid = phase(kwargs['data'], kwargs['sample_support'], kwargs['config_file'], \
                                good_pix = kwargs['good_pix'], solid_known = kwargs['solid_unit'])
    
    # write the h5 file 
    fnam = os.path.join(kwargs['config_file']['output']['path'], 'output.h5')
    utils.io_utils.write_input_output_h5(fnam, data = kwargs['data'], \
            data_retrieved = diff_ret, sample_support = kwargs['sample_support'], \
            sample_support_retrieved = kwargs['sample_support'], good_pix = kwargs['good_pix'], \
            solid_unit = kwargs['solid_unit'], solid_unit_retrieved = solid_ret, modulus_error = emod, \
            fidelity_error = efid)
