import numpy as np
import sys
import os
import ConfigParser

sys.path.append(os.path.abspath('.'))

from utils.disorder      import make_exp
from utils.l2norm        import l2norm
from utils.io_utils      import parse_parameters
from utils.io_utils      import parse_cmdline_args_phasing
from utils.io_utils      import read_input_h5
from utils.io_utils      import write_output_h5

from gpu.phasing.maps import *


def phase(I, solid_support, params, good_pix = None, solid_known = None):
    """
    """
    if good_pix is None :
        good_pix = I > -1
    
    duck = np.random.random(solid_support.shape) + 0.0J
    
    maps = Mappings(duck, support, amp, good_pix, params)
    

if __name__ == "__main__":
    args = parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, solid_known, params = read_input_h5(args.input)
    
    solid_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, solid_known = solid_known)
    
    # write the h5 file 
    #write_output_h5(params['output']['path'], diff, diff_ret, support, \
    #                support, good_pix, solid_known, solid_ret, emod, efid)
