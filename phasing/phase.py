import numpy as np
import sys
import os
import ConfigParser

import crappy_crystals
import crappy_crystals.utils
from crappy_crystals import utils
from crappy_crystals import phasing
from crappy_crystals.phasing.maps import *
from crappy_crystals.phasing.era import ERA


def phase(I, solid_support, params, good_pix = None, solid_known = None):

    solid_ret, info = ERA(I, params['phasing']['era'], solid_support, params, \
                          mask = good_pix, O = solid_known, \
                          background = None, method = 1, hardware = 'cpu', \
                          alpha = 1.0e-10, dtype = 'double', full_output = True)
    
    return solid_ret, info['I'], info['eMod'], np.zeros_like(info['eMod'])

if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, solid_known, params = utils.io_utils.read_input_h5(args.input)
    
    solid_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, solid_known = solid_known)
    
    # write the h5 file 
    utils.io_utils.write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, solid_known, solid_ret, emod, efid)
