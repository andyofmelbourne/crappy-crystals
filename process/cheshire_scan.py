# read in a solid unit, unit-cell params, the space group, the diffraction and the lattice
# then perform a cheshire scan and output the solid unit shifted and oriented
# at the best location.

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

from itertools import product
from functools import reduce

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path = [os.path.join(root, 'utils')] + sys.path

#import io_utils
#import duck_3D
#import forward_sim
#import phasing_3d
#import maps
#import fidelity
import symmetry_operations

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask):
    sym_ops = symmetry_operations.P212121(unit_cell, diff.shape, dtype=np.complex64)
    
    # define the search space
    I = range(sym_ops.Cheshire_cell[0])
    J = range(sym_ops.Cheshire_cell[1])
    K = range(sym_ops.Cheshire_cell[2])
    
    # make the errors
    errors = np.zeros((len(I), len(J), len(K)), dtype=np.float)
    errors.fill(np.inf)
    
    # only evaluate the error on Bragg peaks that are strong 
    Bragg_mask = B > 1.0e-1 * B[0,0,0]
    
    # symmetrise it so that we have all pairs in the point group
    Bragg_mask = sym_ops.solid_syms_Fourier(Bragg_mask, apply_translation=False)
    Bragg_mask = np.sum(Bragg_mask, axis=0)>0

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
      
    # shift
    qi = np.fft.fftfreq(s.shape[0]) 
    qj = np.fft.fftfreq(s.shape[1])
    qk = np.fft.fftfreq(s.shape[2])
    T0 = np.exp(- 2J * np.pi * I[i] * qi)
    T1 = np.exp(- 2J * np.pi * J[j] * qj)
    T2 = np.exp(- 2J * np.pi * K[k] * qk)
    phase_ramp = reduce(np.multiply.outer, [T0, T1, T2])
    s1         = s * phase_ramp
    s1         = np.fft.ifftn(s1)
    return errors[i,j,k], (I[i], J[j], K[k])

def shift_Fourier(s, shift):
    qi = np.fft.fftfreq(s.shape[0]) 
    qj = np.fft.fftfreq(s.shape[1])
    qk = np.fft.fftfreq(s.shape[2])
    T0 = np.exp(- 2J * np.pi * shift[0] * qi)
    T1 = np.exp(- 2J * np.pi * shift[1] * qj)
    T2 = np.exp(- 2J * np.pi * shift[2] * qk)
    phase_ramp = reduce(np.multiply.outer, [T0, T1, T2])
    s1         = s * phase_ramp
    return s1

def shift_Real(s, shift):
    out = np.fft.fftn(s)
    return np.fft.ifftn(shift_Fourier(out, shift))

def calc_pearson(x, y):
    X  = np.sum(x)
    Y  = np.sum(y)
    XY = np.sum(x*y)
    XX = np.sum(x**2)
    YY = np.sum(y**2)
    N  = x.size
    return (N * XY - X*Y)/np.sqrt( (N*XX - X**2)*(N*YY - Y**2))


def chesh_scan_w_flips(diff, unit_cell, sin, D, B, mask, spacegroup = 'P212121'):
    if spacegroup == 'P212121':
        # slices gives us the 8 possibilities
        slices = [(slice(None, None, 2*((rank//4)%2)-1), slice(None, None, 2*((rank//2)%2)-1), slice(None, None, 2*((rank)%2)-1)) for rank in range(8)]
        
        sin = sin[slices]
        error, shift = chesh_scan_P212121(diff, unit_cell, sin, D, B, mask)
        
        args = itertools.izip( itertools.repeat(diff), itertools.repeat(unit_cell), [sin[s] for s in slices], \
                               itertools.repeat(D), itertools.repeat(B), itertools.repeat(mask))
    
        res = pool.map() 
        errors = np.array([i[0] for i in res])
        shifts = np.array([i[1] for i in res])



if __name__ == '__main__':
    f = h5py.File('hdf5/pdb/pdb.h5', 'r')
    diff      = f['forward_model_pdb/data'][()]
    unit_cell = f['forward_model_pdb/unit_cell_pixels'][()]
    #sin       = f['forward_model_pdb/solid_unit'][()]
    support   = f['phase/support'][()]
    sin       = f['phase/solid_unit'][()] * support
    lattice   = f['forward_model_pdb/lattice'][()]
    D         = f['forward_model_pdb/diffuse_weighting'][()]
    B         = f['forward_model_pdb/Bragg_weighting'][()]
    if 'forward_model_pdb/mask' in f :
        mask = f['forward_model_pdb/mask'][()]
    else :
        mask = 1
    f.close()



"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == '__main__':
    f = h5py.File('hdf5/pdb/pdb.h5', 'r')
    diff      = f['forward_model_pdb/data'][()]
    unit_cell = f['forward_model_pdb/unit_cell_pixels'][()]
    #sin       = f['forward_model_pdb/solid_unit'][()]
    support   = f['phase/support'][()]
    sin       = f['phase/solid_unit'][()] * support
    lattice   = f['forward_model_pdb/lattice'][()]
    D         = f['forward_model_pdb/diffuse_weighting'][()]
    B         = f['forward_model_pdb/Bragg_weighting'][()]
    if 'forward_model_pdb/mask' in f :
        mask = f['forward_model_pdb/mask'][()]
    else :
        mask = 1
    f.close()

    slices = (slice(None, None, 2*((rank//4)%2)-1), slice(None, None, 2*((rank//2)%2)-1), slice(None, None, 2*((rank)%2)-1))
    sin = sin[slices]
    error, shift = chesh_scan_P212121(diff, unit_cell, sin, D, B, mask)
    print(rank, shift)
    
    slicess = comm.gather(slices)
    errors  = comm.gather(error)
    shifts  = comm.gather(shift)
    
    if rank == 0 :
        i = np.argmin(errors)
        sout = shift_Real(sin[slicess[i]], shifts[i])
    
        # save
        g = h5py.File('temp.h5')
        if 'solid_unit' in g :
            del g['solid_unit']
        g['solid_unit'] = sout
        
        if 'errors' in g :
            del g['errors']
        g['errors'] = np.array(errors)

        if 'shifts' in g :
            del g['shifts']
        g['shifts'] = np.array(shifts)

        if 'i' in g :
            del g['i']
        g['i'] = i
        g.close()
"""
