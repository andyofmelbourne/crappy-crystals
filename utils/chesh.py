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

import numpy as np
import os, sys

try :
    import itertools.izip as zip
except ImportError :
    pass

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path = [os.path.join(root, 'utils')] + sys.path

import symmetry_operations

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask):
    sym_ops = symmetry_operations.P212121(unit_cell, diff.shape, dtype=np.complex64)
    
    # define the search space
    I = range(sym_ops.Cheshire_cell[0])
    J = range(sym_ops.Cheshire_cell[1])
    K = range(sym_ops.Cheshire_cell[2])
    #I = range(unit_cell[0])
    #J = range(unit_cell[1])
    #K = range(unit_cell[2])
    
    # make the errors
    errors = np.zeros((len(I), len(J), len(K)), dtype=np.float)
    errors.fill(np.inf)
    print('errors.shape',errors.shape)
    
    # only evaluate the error on Bragg peaks that are strong 
    thresh = np.percentile(B[B>0], 90.) 
    #Bragg_mask = B > 1.0e-1 * B[0,0,0]
    Bragg_mask = B > thresh
    
    # keep only one orthant
    s = Bragg_mask.shape
    #Bragg_mask[s[0]//2:, :, :] = False
    #Bragg_mask[:, s[1]//2:, :] = False
    #Bragg_mask[:, :, s[2]//2:] = False
    print(np.sum(Bragg_mask), 'Bragg peaks used for cheshire scan')
    
    # symmetrise it so that we have all pairs in the point group
    #Bragg_mask = sym_ops.solid_syms_Fourier(Bragg_mask, apply_translation=False)
    #Bragg_mask = np.sum(Bragg_mask, axis=0)>0

    #def solid_syms_Fourier_masked(self, solid, i, j, k, apply_translation = True, syms = None):
    
    # propagate
    s  = np.fft.fftn(sin)
    s1 = np.zeros_like(s)
    
    qi = np.fft.fftfreq(s.shape[0]) 
    qj = np.fft.fftfreq(s.shape[1])
    qk = np.fft.fftfreq(s.shape[2])
    qi, qj, qk = np.meshgrid(qi, qj, qk, indexing='ij')
    qi = qi[Bragg_mask]
    qj = qj[Bragg_mask]
    qk = qk[Bragg_mask]
    
    sBragg = s[Bragg_mask]

    amp    = np.sqrt(diff)[Bragg_mask] 
    if mask is not 1 :
        mask = mask[Bragg_mask] 
    
    I_norm = np.sum(mask*amp**2)
    D = D[Bragg_mask]
    B = B[Bragg_mask]

    # store the indices for each orientation of the solid_unit
    sBragg_inds, Ts = sym_ops.make_inds_Ts_masked(Bragg_mask)
    
    # precalculate the diffuse scattering
    modes = sBragg[sBragg_inds] * Ts

    D     = D * np.sum( (modes * modes.conj()).real, axis=0)
    
    for i in I:
        for j in J:
            for k in K:
                phase_ramp = np.exp(- 2J * np.pi * (i * qi + j * qj + k * qk))
                 
                U         = np.sum((sBragg * phase_ramp)[sBragg_inds] * Ts, axis=0)
                #assert(np.allclose(U, U2))

                diff_calc = B * (U * U.conj()).real + D
                
                # Emod
                eMod      = np.sum( mask * ( np.sqrt(diff_calc) - amp )**2 )
                eMod      = np.sqrt( eMod / I_norm )
                
                #eMod = 1 - calc_pearson(mask*diff_calc, mask*amp)/2.

                errors[i - I[0], j - J[0], k - K[0]] = eMod
    
    l = np.argmin(errors)
    i, j, k = np.unravel_index(l, errors.shape)
    print('lowest error at: i, j, k, err', i, j, k, errors[i,j,k])
      
    return errors, (I[i], J[j], K[k])

def chesh_scan_P212121_wrap(x):
    return chesh_scan_P212121(*x)

def chesh_scan_w_flips(diff, unit_cell, sin, D, B, mask, single_thread=True, spacegroup = 'P212121'):
    if spacegroup == 'P212121':
        # slices gives us the 8 possibilities
        slices = [(slice(None, None, 2*((rank//4)%2)-1), slice(None, None, 2*((rank//2)%2)-1), slice(None, None, 2*((rank)%2)-1)) for rank in range(8)]
        slices = slices[::2]
        
        import itertools
        args = zip( itertools.repeat(diff), itertools.repeat(unit_cell), [sin[s].copy() for s in slices], \
                               itertools.repeat(D), itertools.repeat(B), itertools.repeat(mask))
         
        if single_thread :
            res = [chesh_scan_P212121_wrap(arg) for arg in args]
        else :
            from multiprocessing import Pool
            pool = Pool()
            res    = pool.map(chesh_scan_P212121_wrap, args) 
        
        errors = np.array([i[0] for i in res])
        shifts = np.array([i[1] for i in res])
        
        i = np.argmin([np.min(e) for e in errors])
        # shift and permiate the sample and support
        shift = shifts[i]
        error = errors[i]
        sl    = slices[i]
        return shift, error, sl
    
    elif spacegroup == 'P1':
        return np.array([0.]), 0., (slice(None), slice(None), slice(None))
    
    else :
        raise ValueError('spacegroup ' + spacegroup + ' unsupported')
