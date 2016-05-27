import numpy as np
import sys
import os
import ConfigParser

import crappy_crystals
import crappy_crystals.utils
from crappy_crystals import utils
from crappy_crystals import phasing
from crappy_crystals.phasing.maps import *


def phase(I, solid_support, params, good_pix = None, solid_known = None):
    """
    """
    if good_pix is None :
        good_pix = I > -1

    maps = Mappings(params)
    
    def Pmod(x):
        """
        O --> solid_syms
        _Pmod
        modes --> O
        """
        solid_syms = maps.solid_syms(x)
        x = Pmod_(solid_syms[0], I, maps.make_diff(solid_syms = solid_syms), good_pix)
        x = np.fft.ifftn(x)
        return x

    def Psup(x):
        # apply support
        y = x * solid_support
        
        # apply reality
        y.imag = 0.0
        
        # apply positivity
        y[np.where(y<0)] = 0.0
        return y

    #Psup = lambda x : (x * solid_support).real + 0.0J
    
    ERA = lambda x : Psup(Pmod(x))
    HIO = lambda x : HIO_(x.copy(), Pmod, Psup, beta=1.)
    DM  = lambda x : DM_(x, Pmod, Psup, beta=1.0)
    DM_to_sol = lambda x : DM_to_sol_(x, Pmod, Psup, beta=1.0)

    e_mod = []
    e_sup = []
    e_fid = []

    print '\nalg: progress iteration modulus error fidelty'
    x = np.random.random(solid_support.shape) + 0.0J
    x = Psup(x)

    iters = params['phasing']['hio']
    for i in range(iters):
        x = HIO(x)
        
        # calculate the fidelity and modulus error
        M = maps.make_diff(solid = x)
        e_mod.append(utils.l2norm.l2norm(np.sqrt(I), np.sqrt(M)))
        #e_sup.append(l2norm(x, Psup(x)))
        if solid_known is not None :
            e_fid.append(utils.l2norm.l2norm(solid_known + 0.0J, x))
        else :
            e_fid.append(-1)
        
        update_progress(i / max(1.0, float(iters-1)), 'HIO', i, e_mod[-1], e_fid[-1])

    iters = params['phasing']['era']
    for i in range(iters):
        x = ERA(x)
        
        # calculate the fidelity and modulus error
        M = maps.make_diff(solid = x)
        e_mod.append(utils.l2norm.l2norm(np.sqrt(I), np.sqrt(M)))
        e_sup.append(utils.l2norm.l2norm(x, Psup(x)))
        if solid_known is not None :
            e_fid.append(utils.l2norm.l2norm(solid_known + 0.0J, x))
        else :
            e_fid.append(-1)
        
        update_progress(i / max(1.0, float(iters-1)), 'ERA', i, e_mod[-1], e_fid[-1])
    print '\n'

    return x, M, e_mod, e_fid


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, solid_known, params = utils.io_utils.read_input_h5(args.input)
    
    solid_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, solid_known = solid_known)
    
    # write the h5 file 
    utils.io_utils.write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, solid_known, solid_ret, emod, efid)
