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
import h5py

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import io_utils
import duck_3D
import forward_sim
import maps

import symmetry_operations

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=None, Bragg_mask=None, check_finite=False):
    sym_ops  = symmetry_operations.P212121(unit_cell, diff.shape, dtype=np.complex64)

    # propagate
    s  = np.fft.fftn(sin)
    
    # define the real-pace search grid
    if scan_grid is None :
        print(sym_ops.Cheshire_cell)
        I = range(sym_ops.Cheshire_cell[0])
        J = range(sym_ops.Cheshire_cell[1])
        K = range(sym_ops.Cheshire_cell[2])
    else :
        I, J, K = scan_grid

    # testing
    #I, J, K = [range(u) for u in unit_cell]
    
    # make the errors
    errors = np.zeros((len(I), len(J), len(K)), dtype=np.float)
    errors.fill(np.inf)
    print('errors.shape',errors.shape)
    
    if Bragg_mask is None :
        # get the reciprocal lattice points
        plattice = symmetry_operations.make_lattice_subsample(unit_cell, diff.shape, N = None, subsamples=50)
        
        # only evaluate the error on Bragg peaks that are strong 
        Bragg_mask = plattice.copy().astype(np.bool)
        
        sh = Bragg_mask.shape
        Bragg_mask[sh[0]//2:, :, :] = False
        Bragg_mask[:, sh[1]//2:, :] = False
        Bragg_mask[:, :, sh[2]//2:] = False
    
    print(np.sum(Bragg_mask), 'Bragg peaks used for cheshire scan')
    
    # make the qspace grid
    qi = np.fft.fftfreq(s.shape[0]) 
    qj = np.fft.fftfreq(s.shape[1])
    qk = np.fft.fftfreq(s.shape[2])

    # hopefully plattice is separable
    qi, qj, qk = np.meshgrid(qi, qj, qk, indexing='ij')
    qi = np.unique(qi[Bragg_mask])
    qj = np.unique(qj[Bragg_mask])
    qk = np.unique(qk[Bragg_mask])
    
    modes = sym_ops.solid_syms_Fourier(s)
    modes = np.array([m[Bragg_mask] for m in modes])

    amp    = np.sqrt(diff)[Bragg_mask] 
    if mask is not 1 :
        mask = mask[Bragg_mask] 
    
    I_norm = np.sum(mask*amp**2)
    
    Dsub= D[Bragg_mask]
    Bsub= B[Bragg_mask]

    # calculate the diffuse scattering
    Dsub = Dsub * np.sum( (modes * modes.conj()).real, axis=0)
    
    U = np.zeros_like(modes[0])
    for ii, i in enumerate(I):
        prI = np.exp(- 2J * np.pi * (i * qi))
        for jj, j in enumerate(J):
            prJ = np.exp(- 2J * np.pi * (j * qj))
            for kk, k in enumerate(K):
                prK = np.exp(- 2J * np.pi * (k * qk))
                
                U.fill(0)
                U += modes[0] * np.multiply.outer(np.multiply.outer(prI, prJ), prK).ravel()
                U += modes[1] * np.multiply.outer(np.multiply.outer(prI, prJ.conj()), prK.conj()).ravel()
                U += modes[2] * np.multiply.outer(np.multiply.outer(prI.conj(), prJ), prK.conj()).ravel()
                U += modes[3] * np.multiply.outer(np.multiply.outer(prI.conj(), prJ.conj()), prK).ravel()
                 
                diff_calc = Bsub * (U * U.conj()).real + Dsub
                
                # Emod
                eMod      = np.sum( mask * ( np.sqrt(diff_calc) - amp )**2 )
                eMod      = np.sqrt( eMod / I_norm )
                
                #eMod = 1 - calc_pearson(mask*diff_calc, mask*amp)/2.

                errors[ii, jj, kk] = eMod
                #print(i, j, k, eMod)
    
    l = np.argmin(errors)
    i, j, k = np.unravel_index(l, errors.shape)
    i, j, k = I[i], J[j], K[k]
    print('lowest error at: i, j, k, err', i, j, k, errors.ravel()[l])
    
    if check_finite : 
        # now we need to account for finite crystal effects 
        # this means that we should asses the full error with offsets
        # of one cheshire cell and account for interBragg intensities
        I = range(i, i + diff.shape[0], sym_ops.Cheshire_cell[0])
        J = range(j, j + diff.shape[1], sym_ops.Cheshire_cell[1])
        K = range(k, k + diff.shape[2], sym_ops.Cheshire_cell[2])
        Bragg_mask = np.ones_like(Bragg_mask)
        print('I:',list(I))
        print('J:',list(J))
        print('K:',list(K))
        
        Bragg_mask[sh[0]//2:, :, :] = False
        Bragg_mask[:, sh[1]//2:, :] = False
        Bragg_mask[:, :, sh[2]//2:] = False
        errors, (i, j, k) = chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=[I, J, K], Bragg_mask=Bragg_mask, check_finite=False)
        print(errors)
      
    return errors, (i, j, k)


if __name__ == '__main__':
    f = h5py.File('../hdf5/duck/duck.h5', 'r')
    
    # 
    diff = f['forward_model/data'][()]
    uc   = np.array([32,32,32])
    s    = f['forward_model/solid_unit'][()]
    D    = f['forward_model/diffuse_weighting'][()]
    B    = f['forward_model/Bragg_weighting'][()]
    
    """
    # symetry mapper
    sym_ops = symmetry_operations.P212121(uc, diff.shape, dtype=np.complex128)

    m       = sym_ops.solid_syms_Fourier(np.fft.fftn(s))
    diff2 = B*np.abs(np.sum(m, axis=0))**2+D*np.sum(np.abs(m)**2, axis=0)
    print('initial error:', np.sum( (np.sqrt(diff) - np.sqrt(diff2))**2 ))

    errors = np.zeros((4,4), dtype=np.float)
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            s2 = np.roll(s,  i, 0)
            s2 = np.roll(s2, j, 1)

            m     = sym_ops.solid_syms_Fourier(np.fft.fftn(s2))
            diff2 = B*np.abs(np.sum(m, axis=0))**2+D*np.sum(np.abs(m)**2, axis=0)
            
            errors[i,j]  =  np.sum( (np.sqrt(diff) - np.sqrt(diff2))**2 )
            print('shifted error:', i, j, errors[i, j])
    """
    # 131,54s
    s2 = np.roll(s[:, ::-1, ::-1], 0, 0)
    
    #from chesh import chesh_scan_P212121
    errors, i = chesh_scan_P212121(diff, np.array(uc), s2, D, B, 1, check_finite=True)
