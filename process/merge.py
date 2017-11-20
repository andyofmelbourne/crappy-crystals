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
sys.path = [os.path.join(root, 'utils')] + sys.path

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# hack for import error in pyximport
for r in range(size):
    if rank == r :
        import maps
    comm.barrier()

import io_utils
import duck_3D
import forward_sim
import phasing_3d
import fidelity

from phase_mpi import *
import glob

def post_align(O1, O2):
    o1 = np.fft.rfftn(O1.real)
    o2 = np.fft.rfftn(O2.real)
    
    # make the qspace grid
    qi = np.fft.fftfreq(O1.shape[0])
    qj = np.fft.fftfreq(O1.shape[1])
    qk = np.fft.fftfreq(O1.shape[2])[:O1.shape[2]/2 + 1]
    
    I = np.linspace(-1, 1, 11)
    J = np.linspace(-1, 1, 11)
    K = np.linspace(-1, 1, 11)
    fids = np.zeros( (len(I), len(J), len(K)), dtype=np.float)
    for ii, i in enumerate(I):
        prI = np.exp(- 2J * np.pi * (i * qi))
        for jj, j in enumerate(J):
            prJ = np.exp(- 2J * np.pi * (j * qj))
            for kk, k in enumerate(K):
                prK = np.exp(- 2J * np.pi * (k * qk))
                
                phase_ramp = np.multiply.outer(np.multiply.outer(prI, prJ), prK)
                fids[ii, jj, kk] = calc_fid(o1, o2 * phase_ramp)
    
    l = np.argmin(fids)
    i, j, k = np.unravel_index(l, fids.shape)
    print('lowest error at:', i,j,k, fids[i,j,k])
    i, j, k = I[i], J[j], K[k]
    
    prI = np.exp(- 2J * np.pi * (i * qi))
    prJ = np.exp(- 2J * np.pi * (j * qj))
    prK = np.exp(- 2J * np.pi * (k * qk))
    phase_ramp = np.multiply.outer(np.multiply.outer(prI, prJ), prK)
    
    return np.fft.irfftn(o2 * phase_ramp)

if __name__ == '__main__':
    dirnam = sys.argv[1]
    
    # get the file list
    flist = glob.glob(os.path.join(dirnam, 'O_[0-9]*.h5'))
    
    f   = h5py.File(flist[rank], 'r')
    O   = f['solid_unit'][()]
    if 'eMod' not in f:
        print(rank, flist[rank], [k for k in f.keys()])
    err = f['eMod'][-1]
    f.close()

    # get the rank of the best retrieval
    errs = comm.allgather(err)
    
    # get everyone to align their object with respect to the first
    O0 = comm.bcast(O, root=np.argmin(errs))

    print('merging Os...', rank)
    O  = align_Os(O0, O)
    Os = np.empty((size,)+O.shape, dtype=O.dtype)
    comm.Gather(O, Os, root=0) 

    if rank == 0: 
        O = np.mean([Os[i] for i in np.where(errs < np.mean(errs))[0]], axis=0)

        # output 
        f = h5py.File(os.path.join(dirnam, 'O_merge.h5'))
    
        key = '/solid_unit'
        if key in f :
            del f[key]
        f[key] = O
    
        key = '/errs'
        if key in f :
            del f[key]
        f[key] = np.array(errs)
        f.close()
