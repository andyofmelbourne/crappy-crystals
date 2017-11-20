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
import sys, os
from itertools import product
from functools import reduce

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

cuda_stream = drv.Stream()

    #x = x
    #x = 0.5 + x, 0.5 - y, -z
    #x = -x, -0.5 + y, 0.5 - z # Note: 0.5 + y --> -0.5 + y
    #x = 0.5 - x, -y, 0.5 + z
gpu_fns = SourceModule("""
#include <pycuda-complex.hpp>
#include <stdio.h>

// pythonic mod: https://stackoverflow.com/a/4003287
__device__ int mod (int a, int b)
{
   if(b < 0) //you can check for b == 0 separately and do what you want
     return mod(a, -b);   
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}

__global__ void make_U(int ux, int uy, int uz, 
                       int n, int m, int l, 
                       int x, int y, int z, 
                       pycuda::complex<double> *U, pycuda::complex<double> *O)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int i,j,k;      // O coords
    int i2,j2,k2;   // mapped O coords
    for (int ii = index; ii < (n*m*l); ii += stride){
        // x, y, z
        i =  ii / (m*l);
        j = (ii - i*m*l) / l;
        k = (ii - i*m*l - j*l);
        
        // shifted x, y, z
        i2 = mod((i - x), n);
        j2 = mod((j - y), m);
        k2 = mod((k - z), l);
        U[ii] = O[i2*m*l + j2*l + k2];
        
        // 1/2 + x, 1/2 - y, -z
        i2 = mod((ux/2 + i - x), n);
        j2 = mod((uy/2 - j - y), m);
        k2 = mod((     - k - z), l);
        U[ii] += O[i2*m*l + j2*l + k2];
        
        // -x, 1/2 + y, 1/2 - z
        i2 = mod((     - i - x), n);
        j2 = mod((uy/2 + j - y), m);
        k2 = mod((uz/2 - k - z), l);
        U[ii] += O[i2*m*l + j2*l + k2];
        
        // 1/2 - x, - y, 1/2 + z
        i2 = mod((ux/2 - i - x), n);
        j2 = mod((     - j - y), m);
        k2 = mod((uz/2 + k - z), l);
        U[ii] += O[i2*m*l + j2*l + k2];
    }
}

__global__ void make_M(int ux, int uy, int uz, 
                       int n, int m, int l, 
                       pycuda::complex<double> *M, pycuda::complex<double> *O)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int i,j,k;      // O coords
    int i2,j2,k2;   // mapped O coords
    for (int ii = index; ii < (n*m*l); ii += stride){
        // x, y, z
        i =  ii / (m*l);
        j = (ii - i*m*l) / l;
        k = (ii - i*m*l - j*l);
        
        // x, y, z
        i2 = mod((i), n);
        j2 = mod((j), m);
        k2 = mod((k), l);
        M[ii] = O[i2*m*l + j2*l + k2];
        
        // inv: 1/2 + x, 1/2 - y, -z
        i2 = mod((ux/2 + i), n);
        j2 = mod((uy/2 - j), m);
        k2 = mod((     - k), l);
        M[ii + (n*m*l)] = O[i2*m*l + j2*l + k2];
        
        // -x, 1/2 + y, 1/2 - z
        i2 = mod((     - i), n);
        j2 = mod((uy/2 + j), m);
        k2 = mod((uz/2 - k), l);
        M[ii + 2*(n*m*l)] = O[i2*m*l + j2*l + k2];
        
        // 1/2 - x, - y, 1/2 + z
        i2 = mod((ux/2 - i), n);
        j2 = mod((     - j), m);
        k2 = mod((uz/2 + k), l);
        M[ii + 3*(n*m*l)] = O[i2*m*l + j2*l + k2];
    }
}

__global__ void make_U_F(int ux, int uy, int uz, 
                     int n, int m, int l, 
                     int x, int y, int z, 
                     pycuda::complex<double> *U, pycuda::complex<double> *M)
{
    int i, j, k;
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    pycuda::complex<double> pi2i(0, 6.28318530718);
    
    for (int ii = index; ii < (n*m*l); ii += stride){
        // x, y, z
        i =  ii / (m*l);
        j = (ii - i*m*l) / l;
        k = (ii - i*m*l - j*l);
        
        // shifted x, y, z
        U[ii] = M[ii] * pycuda::exp(-pi2i * (double(x*i)/double(n) + double(y*j)/double(m) + double(z*k)/double(l)));
        
        // 1/2 + x, 1/2 - y, -z
        U[ii] += M[ii +   (n*m*l)] * pycuda::exp(-pi2i * (double(x*i)/double(n) + double(-y*j)/double(m) + double(-z*k)/double(l)));

        // -x, 1/2 + y, 1/2 - z
        U[ii] += M[ii + 2*(n*m*l)] * pycuda::exp(-pi2i * (double(-x*i)/double(n) + double(y*j)/double(m) + double(-z*k)/double(l)));
        
        // 1/2 - x, - y, 1/2 + z
        U[ii] += M[ii + 3*(n*m*l)] * pycuda::exp(-pi2i * (double(-x*i)/double(n) + double(-y*j)/double(m) + double(z*k)/double(l)));
    }
}


__global__ void err_calc(int n, int m, int l, 
                         int x, int y, int z, 
                         pycuda::complex<double> *M,
                         pycuda::complex<double> *Delta,
                         int *mask,
                         double *BW, double *D, double *amp)
{
    int i, j, k;
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    pycuda::complex<double> pi2i(0, 6.28318530718);
    pycuda::complex<double> U;
    
    for (int ii = index; ii < (n*m*l); ii += stride){
        // x, y, z
        i =  ii / (m*l);
        j = (ii - i*m*l) / l;
        k = (ii - i*m*l - j*l);
        if (mask[ii] == 1){
        
        // shifted x, y, z
        U = M[ii] * pycuda::exp(-pi2i * 
                  (double(x*i)/double(n) + double(y*j)/double(m) + double(z*k)/double(l)));
        
        // 1/2 + x, 1/2 - y, -z
        U += M[ii +   (n*m*l)] * pycuda::exp(-pi2i * 
                  (double(x*i)/double(n) + double(-y*j)/double(m) + double(-z*k)/double(l)));

        // -x, 1/2 + y, 1/2 - z
        U += M[ii + 2*(n*m*l)] * pycuda::exp(-pi2i * 
                  (double(-x*i)/double(n) + double(y*j)/double(m) + double(-z*k)/double(l)));
        
        // 1/2 - x, - y, 1/2 + z
        U += M[ii + 3*(n*m*l)] * pycuda::exp(-pi2i * 
                  (double(-x*i)/double(n) + double(-y*j)/double(m) + double(z*k)/double(l)));
        
        Delta[ii] = pow(amp[ii] - sqrt(D[ii] + 
                               BW[ii] * (pow(pycuda::real(U), 2.) + pow(pycuda::imag(U), 2.))), 2.);
        }
    }
}

""")

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path = [os.path.join(root, 'utils')] + sys.path

import symmetry_operations

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=None, Bragg_mask=None, check_finite=False):
    sym_ops  = symmetry_operations.P212121(unit_cell, diff.shape, dtype=np.complex128)
    
    # propagate
    s  = np.fft.fftn(sin)
    
    # define the real-pace search grid
    if scan_grid is None :
        I = np.arange(sym_ops.Cheshire_cell[0], dtype=np.int32)
        J = np.arange(sym_ops.Cheshire_cell[1], dtype=np.int32)
        K = np.arange(sym_ops.Cheshire_cell[2], dtype=np.int32)
    else :
        I, J, K = scan_grid
        I = np.array(I, dtype=np.int32)
        J = np.array(J, dtype=np.int32)
        K = np.array(K, dtype=np.int32)
    
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
    
    modes  = sym_ops.solid_syms_Fourier(s)
    amp    = np.sqrt(diff)
    I_norm = np.sum(mask*amp**2)
    
    # send stuff to the gpu
    err_calc  = gpu_fns.get_function('err_calc')
    err_calc.prepare([np.int32, np.int32, np.int32, np.int32, np.int32, np.int32,
                      np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
    Mg     = gpuarray.to_gpu(np.ascontiguousarray(modes))
    maskg  = gpuarray.to_gpu(np.ascontiguousarray(Bragg_mask.astype(np.int32)))
    BWg    = gpuarray.to_gpu(np.ascontiguousarray(B.astype(np.float64)))
    Dg     = gpuarray.to_gpu(np.ascontiguousarray(
                            (D * np.sum( (modes * modes.conj()).real, axis=0)).astype(np.float64)))
    ampg   = gpuarray.to_gpu(np.ascontiguousarray(amp.astype(np.float64)))
    Deltag = gpuarray.zeros(diff.shape, dtype=np.float64)
    
    shape = np.array(diff.shape, dtype=np.int32)

    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            for kk, k in enumerate(K):
                err_calc.prepared_call( (128,1), (1024,1,1),
                         shape[0], shape[1], shape[2],
                         i, j, k,
                         Mg.gpudata, Deltag.gpudata, maskg.gpudata, BWg.gpudata, Dg.gpudata, ampg.gpudata)
                
                # Emod
                eMod      = gpuarray.sum( Deltag ).get()
                eMod      = np.sqrt( eMod / I_norm )
                
                errors[ii, jj, kk] = eMod
        print(i, j, k, eMod)
    
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





import h5py
f = h5py.File('hdf5/cuda/cuda.h5', 'r')
O = f['forward_model/solid_unit'][()]
BW = f['/forward_model/Bragg_weighting'][()]
DW = f['/forward_model/diffuse_weighting'][()]
diff = f['/forward_model/data'][()]
f.close()

O2 = np.roll( O, (1,-2,3), (0,1,2))
errors, (i,j,k) = chesh_scan_P212121(diff, np.array([64,64,64]), O2, 
                                         DW, BW, 1, check_finite=True)#, scan_grid = [[0],[0],[0]])
"""
#O = np.random.random((4,4,4)) + 0J
Og  = gpuarray.to_gpu(np.ascontiguousarray(O))
Ohg = gpuarray.to_gpu(np.ascontiguousarray(np.fft.fftn(O)))
Ug  = gpuarray.zeros(O.shape, O.dtype)
Uhg = gpuarray.zeros(O.shape, O.dtype)
Mg  = gpuarray.zeros((4,) + O.shape, O.dtype)

make_M     = gpu_fns.get_function('make_M')
make_U     = gpu_fns.get_function('make_U')
make_U_F   = gpu_fns.get_function('make_U_F')
make_Udiff = gpu_fns.get_function('make_Udiff')

s = np.array(O.shape, dtype=np.int32)
t = np.array([0,0,0], dtype=np.int32)
u = s // 2

# make the unit-cell in real-space with the shift
#------------------------------------------------
make_U(u[0], u[1], u[2], s[0], s[1], s[2], t[0], t[1], t[2], Ug, Og, block=(128,1,1), grid=(128,1))
U_real = Ug.get()

# make the unit-cell in Fourier-space with the shift
#---------------------------------------------------
make_M(u[0], u[1], u[2], s[0], s[1], s[2], Mg, Og, block=(1024,1,1), grid=(128,1))
M  = np.fft.fftn(Mg.get(), axes=(1,2,3))
Mg = gpuarray.to_gpu(np.ascontiguousarray(M))

make_U_F(u[0], u[1], u[2], s[0], s[1], s[2], t[0], t[1], t[2], Uhg, Mg, block=(1024,1,1), grid=(128,1))
U_Fourier = np.fft.ifftn(np.ascontiguousarray(Uhg.get()))
print(np.allclose(U_Fourier, U_real))

BWg    = gpuarray.to_gpu(np.ascontiguousarray(BW))
Deltag = gpuarray.zeros(I.shape, I.dtype)
Ig     = gpuarray.to_gpu(np.ascontiguousarray(I - DW * np.sum(np.abs(M)**2, axis=0)))


err = []
for x in range(32):
    print(x)
    for y in range(32):
        for z in range(32):
            t[:] = x, y, z
            make_Udiff(u[0], u[1], u[2],
                       s[0], s[1], s[2],
                       t[0], t[1], t[2],
                       Mg, Deltag, BWg, Ig,
                       block = (1024,1,1), grid=(128,1))
            err.append(gpuarray.sum(Deltag).get())
            #Delta = (I - DW * np.sum(np.abs(M)**2, axis=0) - BW * np.abs(np.sum(M, axis=0))**2)**2
            #print(np.allclose(Deltag.get()[:64,:64,:64], Delta[:64,:64,:64]))
"""
