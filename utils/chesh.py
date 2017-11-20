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

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=None, Bragg_mask=None, check_finite=False):
    sym_ops  = symmetry_operations.P212121(unit_cell, diff.shape, dtype=np.complex64)

    # propagate
    s  = np.fft.fftn(sin)
    
    # define the real-pace search grid
    if scan_grid is None :
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
        masksub = mask[Bragg_mask] 
    else :
        masksub = 1
    
    I_norm = np.sum(masksub*amp**2)
    
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
                eMod      = np.sum( masksub * ( np.sqrt(diff_calc) - amp )**2 )
                eMod      = np.sqrt( eMod / I_norm )
                
                #eMod = 1 - calc_pearson(mask*diff_calc, mask*amp)/2.

                errors[ii, jj, kk] = eMod
                #print(i, j, k, eMod)
    
    l = np.argmin(errors)
    i, j, k = np.unravel_index(l, errors.shape)
    i, j, k = I[i], J[j], K[k]
    
    if check_finite : 
        # now we need to account for finite crystal effects 
        # this means that we should asses the full error with offsets
        # of one cheshire cell and account for interBragg intensities
        I = range(i, i + diff.shape[0], sym_ops.Cheshire_cell[0])
        J = range(j, j + diff.shape[1], sym_ops.Cheshire_cell[1])
        K = range(k, k + diff.shape[2], sym_ops.Cheshire_cell[2])
        Bragg_mask = np.ones_like(Bragg_mask)
        
        Bragg_mask[sh[0]//2:, :, :] = False
        Bragg_mask[:, sh[1]//2:, :] = False
        Bragg_mask[:, :, sh[2]//2:] = False
        errors, (i, j, k) = chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=[I, J, K], Bragg_mask=Bragg_mask, check_finite=False)
      
    return errors, (i, j, k)

def chesh_scan_P212121_wrap(x):
    return chesh_scan_P212121(*x, check_finite=True)

def chesh_scan_w_flips(diff, unit_cell, sin, D, B, mask, single_thread=True, spacegroup = 'P212121'):
    if spacegroup == 'P212121':
        # slices gives us the 8 possibilities
        slices = [(slice(None, None, 2*((rank//4)%2)-1), slice(None, None, 2*((rank//2)%2)-1), slice(None, None, 2*((rank)%2)-1)) for rank in range(8)]
        #slices = slices[::2]
        slices = [slices[0], ]
        
        import itertools
        args = zip( itertools.repeat(diff), itertools.repeat(unit_cell), [sin[s].copy() for s in slices], \
                               itertools.repeat(D), itertools.repeat(B), itertools.repeat(mask))
         
        if single_thread :
            res = [chesh_scan_P212121_wrap(arg) for arg in args]
        else :
            from multiprocessing import Pool
            print('using multiprocessing')
            pool = Pool()
            res    = pool.map(chesh_scan_P212121_wrap, args) 
        
        errors = np.array([i[0] for i in res])
        shifts = np.array([i[1] for i in res])
        
        i = np.argmin([np.min(e) for e in errors])
        # shift and permiate the sample and support
        shift = shifts[i]
        error = errors[i]
        sl    = slices[i]
        print('sl:', sl, rank)
        return shift, error, sl
    
    elif spacegroup == 'P1':
        return np.array([0.]), 0., (slice(None), slice(None), slice(None))
    
    else :
        raise ValueError('spacegroup ' + spacegroup + ' unsupported')
