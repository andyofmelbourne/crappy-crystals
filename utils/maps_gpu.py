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

#import symmetry_operations 
import padding
import add_noise_3d
import io_utils

import pyximport; pyximport.install()
#from ellipse_2D_cython import project_2D_Ellipse_arrays_cython
from ellipse_2D_cython_new import project_2D_Ellipse_arrays_cython_test

import phasing_3d
from phasing_3d.src.mappers import Modes
from phasing_3d.src.mappers import isValid

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

import ellipse_2D_cuda 
cuda_stream = drv.Stream()

# WARNING 
# the map function may encounter a race condition
"""
calculates:
modes[jj] = O[ii]
for len(ii) = len(jj) = n
"""
gpu_fns = SourceModule("""
#include <pycuda-complex.hpp>
__global__ void map(int n, pycuda::complex<double> *O, pycuda::complex<double> *modes, unsigned int *ii, unsigned int *jj)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        modes[jj[i]] = O[ii[i]];
    }
}

__global__ void imap(int n, int m, pycuda::complex<double> *O, pycuda::complex<double> *modes, unsigned int *ii, unsigned int *jj)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // prevent the race condition
    for (int j = 0; j < m; j++){
    for (int i = index + j*n; i < (j+1)*n; i += stride){
         O[ii[i]] += (1./double(m)) * modes[jj[i]] ;
    }
    }
}

__global__ void make_U(int n, int m, pycuda::complex<double> *U, pycuda::complex<double> *modes)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < n; i++){
    for (int j = index; j < m; j += stride){
        U[j] = U[j] + modes[m*i+j];
    }
    }
}

__global__ void make_I(int m, pycuda::complex<double> *modes, double *DW, double *BW, double *I)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double r, i;
    for (int j = index; j < m; j += stride){
        //I[j] = BW[j]*pycuda::abs(modes[j]+modes[m + j]+modes[2*m + j]+modes[3*m+j])*pycuda::abs(modes[j]+modes[m + j]+modes[2*m + j]+modes[3*m+j]) + DW[j]*(pycuda::abs(modes[j])*pycuda::abs(modes[j])+pycuda::abs(modes[m + j])*pycuda::abs(modes[m + j])+pycuda::abs(modes[2*m + j])*pycuda::abs(modes[2*m + j])+pycuda::abs(modes[3*m+j])*pycuda::abs(modes[3*m+j]));
        r = pycuda::real(modes[j]+modes[m + j]+modes[2*m + j]+modes[3*m+j]);
        i = pycuda::imag(modes[j]+modes[m + j]+modes[2*m + j]+modes[3*m+j]);
        I[j] = BW[j]*(r*r + i*i);
        
        r = pow(pycuda::real(modes[j]), 2) + pow(pycuda::real(modes[m + j]), 2) + pow(pycuda::real(modes[2*m + j]), 2) + pow(pycuda::real(modes[3*m + j]), 2);
        i = pow(pycuda::imag(modes[j]), 2) + pow(pycuda::imag(modes[m + j]), 2) + pow(pycuda::imag(modes[2*m + j]), 2) + pow(pycuda::imag(modes[3*m + j]), 2);
        I[j]+= DW[j]*(r + i);
    }
}

__global__ void make_xy(int m, int n, pycuda::complex<double> *modes, double *x, double *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride){
        x[i] = sqrt(pycuda::abs(modes[i]) * pycuda::abs(modes[i]));
        
        y[i] = 0.;
        for (int j = 1; j < m; j++) {
            y[i] += pycuda::abs(modes[n*j + i]) * pycuda::abs(modes[n*j + i]);
        }
        y[i] = sqrt(y[i]);
    }
    
}

__global__ void modes_xy(int m, int n, pycuda::complex<double> *modes, double *x, double *yp, double *y)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride){
        modes[i] = pycuda::polar(x[i], pycuda::arg(modes[i]));
        
        for (int j = 1; j < m; j++) {
            modes[n*j + i] = modes[n*j + i] * yp[i] / (y[i] + 1.0e-16);
        }
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

class P212121():
    """
    Store arrays to make the crystal mapping more
    efficient.

    Assume that Fourier space arrays are fft shifted.

    Perform symmetry operations with the np.fft.fftfreq basis
    so that (say) a flip operation behaves like:
    a         = [0, 1, 2, 3, 4, 5, 6, 7]
    a flipped = [0, 7, 6, 5, 4, 3, 2, 1]

    or:
    i         = np.fft.fftfreq(8)*8
              = [ 0,  1,  2,  3, -4, -3, -2, -1]
    i flipped = [ 0, -1, -2, -3, -4,  3,  2,  1]

    symmetry related points:
    
    x = x
    x = 0.5 + x, 0.5 - y, -z
    x = -x, -0.5 + y, 0.5 - z # Note: 0.5 + y --> -0.5 + y
    x = 0.5 - x, -y, 0.5 + z
    """
    spacegroup = 'P212121'
    
    def __init__(self, unitcell_size, det_shape, sup=None):
        # only calculate the translations when they are needed
        self.no_solid_units = np.int32(4)
        
        self.Pat_sym_ops   = np.int32(8)
        self.unitcell_size = unitcell_size
        self.Cheshire_cell = tuple(np.rint(self.unitcell_size/2.).astype(np.int))
        self.det_shape     = det_shape
        
        self.shape = ((self.no_solid_units,) + det_shape)
        
        self.update_sup(np.ones_like(sup))
        self.iig_all = self.iig.copy()
        self.jjg_all = self.jjg.copy()
        self.update_sup(sup)
        
        self.map  = gpu_fns.get_function("map")
        self.imap = gpu_fns.get_function("imap")

    def update_sup(self, sup):
        # make the mapping indices:
        ###########################
        s = self.shape
        
        if sup is None :
            self.sup = np.ones(self.shape[1:], dtype=np.bool)
        else :
            self.sup = sup.copy()
        
        x = np.fft.fftfreq(s[1], 1/s[1]).astype(np.uint32)
        y = np.fft.fftfreq(s[2], 1/s[2]).astype(np.uint32)
        z = np.fft.fftfreq(s[3], 1/s[3]).astype(np.uint32)
        x, y, z = np.meshgrid(x,y,z, indexing='ij')
        
        x,y,z = x[sup], y[sup], z[sup]
        N  = s[1]*s[2]*s[3]
        #ii = np.hstack( tuple([s[2]*s[3]*(x%s[1]) + s[3]*(y%s[2]) + z%s[3] for i in range(2)]) )  
        #jj = np.hstack( (s[2]*s[3]*(           x  % s[1]) + s[3]*(           y  % s[2]) +            z  % s[3],
        #                 s[2]*s[3]*((s[1]//2 + x) % s[1]) + s[3]*((s[2]//2 - y) % s[2]) +          (-z) % s[3] + N))
        ii = np.hstack( tuple([s[2]*s[3]*(x%s[1]) + s[3]*(y%s[2]) + z%s[3] for i in range(s[0])]) )  
        
        # these are the 1D (raveled) indices for the four symettry related solids
        u = self.unitcell_size
        jj = np.hstack( (s[2]*s[3]*(            x  % s[1]) + s[3]*(            y  % s[2]) +             z  % s[3],
                         s[2]*s[3]*(( u[0]//2 + x) % s[1]) + s[3]*(( u[1]//2 - y) % s[2]) +           (-z) % s[3] + N,
                         s[2]*s[3]*((         - x) % s[1]) + s[3]*((-u[1]//2 + y) % s[2]) + ( u[2]//2 - z) % s[3] + 2*N,
                         s[2]*s[3]*(( u[0]//2 - x) % s[1]) + s[3]*((         - y) % s[2]) + ( u[2]//2 + z) % s[3] + 3*N) )
        self.N   = np.int32(len(ii) / self.no_solid_units)
        self.iig = gpuarray.to_gpu(np.ascontiguousarray(ii, np.uint32))
        self.jjg = gpuarray.to_gpu(np.ascontiguousarray(jj, np.uint32))
        

    
    def solid_syms_real(self, solidg, symsg):
        """
        This uses pixel shifts (not phase ramps) for translation.
        Therefore sub-pixel shifts are ignored.
        """
        symsg.fill(0)
        grid  = (int(round(self.N/32.))+1, 1)
        self.map(np.int32(self.N * self.no_solid_units), solidg, symsg, self.iig, self.jjg, block=(32,1,1), grid=grid)
        return symsg
    
    def av_solid_real(self, solidg, symsg, allpix = False):
        """
        This uses pixel shifts (not phase ramps) for translation.
        Therefore sub-pixel shifts are ignored.
        """
        solidg.fill(0)
        grid  = (int(round(self.N/32.))+1, 1)
        if allpix :
            self.imap(np.int32(self.sup.size), np.int32(self.no_solid_units), solidg, symsg, self.iig_all, self.jjg_all, block=(32,1,1), grid=grid)
        else :
            self.imap(self.N, np.int32(self.no_solid_units), solidg, symsg, self.iig, self.jjg, block=(32,1,1), grid=grid)
        return solidg


class Mapper_ellipse():
    """
    Only supports P212121 for now
    """

    def __init__(self, I, **args):
        # dtype
        #-----------------------------------------------
        if isValid('dtype', args) :
            dtype = args['dtype']
            a       = np.array([1], dtype=dtype)
            c_dtype = np.array(a+1J).dtype
        else :
            dtype   = np.float64
            c_dtype = np.complex128
        
        # initialise the object
        #-----------------------------------------------
        if isValid('solid_unit', args):
            O = args['solid_unit'].astype(c_dtype)
        else :
            print('initialising object with random numbers')
            O = np.random.random(I.shape).astype(c_dtype)
            print(O.shape)

        # diffuse and Bragg weightings
        #-----------------------------
        if isValid('Bragg_weighting', args):
            self.unit_cell_weighting = args['Bragg_weighting']
        else :
            self.unit_cell_weighting = np.zeros_like(I)
        
        if isValid('diffuse_weighting', args):
            self.diffuse_weighting   = args['diffuse_weighting']
        else :
            self.diffuse_weighting   = np.zeros_like(I)
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = np.ones(I.shape, dtype=np.bool)
        
        self.I_norm = I.sum()
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']
        
        self.amp    = np.sqrt(I.astype(dtype))
        
        # define the support projection
        #-----------------------------------------------
        if isValid('support', args) :
            self.support = args['support']
        else :
            self.support = 1
        
        if isValid('support_update_freq', args) :
            self.support_update_freq = args['support_update_freq']
        else :
            self.support_update_freq = 1
        
        if isValid('voxels', args) :
            self.voxel_number  = args['voxels']
            self.voxel_support = np.ones(O.shape, dtype=np.bool)
            
            intensity = (O * O.conj()).real.astype(np.float64)
            print('number of pixels greater than zero in input:', np.sum(intensity>0))
            self.voxel_support = choose_N_highest_pixels(intensity, self.voxel_number, support = self.support)
        else :
            self.voxel_number  = False
            self.voxel_support = self.support.copy()
        
        if isValid('overlap', args) :
            self.overlap = args['overlap']
        else :
            self.overlap = None
        
        if isValid('voxel_sup_blur', args) :
            self.voxel_sup_blur = args['voxel_sup_blur']
        else :
            self.voxel_sup_blur = None
        
        if isValid('voxel_sup_blur_frac', args) :
            self.voxel_sup_blur_frac = args['voxel_sup_blur_frac']
        else :
            self.voxel_sup_blur_frac = None
        
        if self.voxel_sup_blur_frac is not None :
            print('\n\nvoxel_sup_blur is not None...')

        # make the crystal symmetry operator
        #-----------------------------------
        self.sym_ops = P212121(args['unit_cell'], I.shape, sup=self.voxel_support)
        
        # initialise gpu arrays
        #-----------------------------------------------
        self.O     = gpuarray.to_gpu(np.ascontiguousarray(O))
        self.Ug     = gpuarray.zeros(O.shape, O.dtype)
        self.ampg   = gpuarray.to_gpu(np.ascontiguousarray(self.amp))
        self.modes = gpuarray.zeros(self.sym_ops.shape, O.dtype) 
        self.DWg = gpuarray.to_gpu(np.ascontiguousarray(self.diffuse_weighting))
        self.BWg = gpuarray.to_gpu(np.ascontiguousarray(self.unit_cell_weighting))
        self.Ig  = gpuarray.empty(I.shape, I.dtype)
        self.xg  = gpuarray.empty(I.shape, I.dtype)
        self.yg  = gpuarray.empty(I.shape, I.dtype)
        self.xpg  = gpuarray.empty(I.shape, I.dtype)
        self.ypg  = gpuarray.empty(I.shape, I.dtype)
        
        # make the reconstruction modes
        #------------------------------
        self.modes = self.sym_ops.solid_syms_real(self.O, self.modes)
        #self.O      = self.object(self.modes)
        
        # FFTs
        #------------------------------
        from reikna.cluda import cuda_api
        import reikna.fft
        api = cuda_api()
        thr = api.Thread(pycuda.autoinit.context)
        fftM3D       = reikna.fft.FFT(self.modes, axes=(1,2,3))
        fftM4D       = reikna.fft.FFT(self.modes, axes=(0,1,2,3))
        fftO3D       = reikna.fft.FFT(self.O, axes=(0,1,2))
        self.fftM3Dc = fftM3D.compile(thr, fast_math=False)
        self.fftM4Dc = fftM4D.compile(thr, fast_math=False)
        self.fftO3Dc = fftO3D.compile(thr, fast_math=False)

        self._make_I  = gpu_fns.get_function("make_I")
        self._make_U  = gpu_fns.get_function("make_U")
        #self._make_xy = ellipse_2D_cuda.gpu_fns.get_function("make_xy")
        #self._modes_xy= ellipse_2D_cuda.gpu_fns.get_function("modes_xy")
        self._make_xy = gpu_fns.get_function("make_xy")
        self._modes_xy= gpu_fns.get_function("modes_xy")
        self._project_2D_Ellipse_arrays_cuda = \
                         ellipse_2D_cuda.gpu_fns.get_function("project_2D_Ellipse_arrays_cuda")
        #self.fftM3Dc(self.modes, self.modes)
        
        # precalculate the ellipse projection arguments
        #----------------------------------------------
        self.Wx         = (self.diffuse_weighting + self.sym_ops.no_solid_units * self.unit_cell_weighting)
        self.Wy         = self.diffuse_weighting
        self.mask       = np.ones(O.shape, dtype=np.uint8)
        self.I_ravel    = I.astype(dtype)
        
        self.I0g   = gpuarray.to_gpu(np.ascontiguousarray(I))
        self.Wxg   = gpuarray.to_gpu(np.ascontiguousarray(self.Wx))
        self.Wyg   = gpuarray.to_gpu(np.ascontiguousarray(self.Wy))
        self.maskg = gpuarray.to_gpu(np.ascontiguousarray(self.mask))
    
        print('EMOD input:', self.Emod(self.modes.copy()))
        
    def next_iter(self, modes, iter):
        # finite support
        if self.voxel_number and (iter % self.support_update_freq == 0) :
            # do one Pmod 
            modes = self.Pmod(modes)
            self.O = self.sym_ops.av_solid_real(self.O, modes, allpix=True)
            
            O = self.O.get()

            # blur O
            # bias low angle scatter for voxel support update
            if self.voxel_sup_blur is not None and self.voxel_sup_blur > 0.01 :
                print('\n\nbluring sample...')
                import scipy.ndimage.filters
                from scipy.ndimage.filters import gaussian_filter
                O = gaussian_filter(O.real, self.voxel_sup_blur, mode='wrap') + 1J * gaussian_filter(O.imag, self.voxel_sup_blur, mode='wrap')
                
            if self.voxel_sup_blur_frac is not None and self.voxel_sup_blur > 0.01 :
                self.voxel_sup_blur *= self.voxel_sup_blur_frac
                print('\n\nblur factor...', self.voxel_sup_blur, self.voxel_sup_blur_frac)
            
    
            intensity = (O * O.conj()).real
            modes = self.sym_ops.solid_syms_real(gpuarray.to_gpu(np.ascontiguousarray(O)), modes)
            
            U = modes.get()
            U = (U * U.conj()).real
            
            # make the crystal
            Crystal = []
            for i in [0, self.sym_ops.unitcell_size[0]]:
                for j in [0, self.sym_ops.unitcell_size[1]]:
                    for k in [0, self.sym_ops.unitcell_size[2]]:
                        for ii in range(U.shape[0]):
                            Crystal.append(np.roll(U[ii], (i,j,k), (0,1,2)))
            
            syms = np.array(Crystal)
            
            """
            # bias low angle scatter for voxel support update
            if self.voxel_sup_blur is not None and self.voxel_sup_blur > 0.01 :
                print('\n\nbluring sample...')
                import scipy.ndimage.filters
                from scipy.ndimage.filters import gaussian_filter
                intensity = gaussian_filter(intensity, self.voxel_sup_blur, mode='wrap')
                #voxel_number_temp = int((1+self.voxel_sup_blur) * self.voxel_number)
                # testing
                voxel_number_temp = self.voxel_number
            else :
                voxel_number_temp = self.voxel_number
            
            if self.voxel_sup_blur_frac is not None and self.voxel_sup_blur > 0.01 :
                self.voxel_sup_blur *= self.voxel_sup_blur_frac
                print('\n\nnew blur sigma value...', self.voxel_sup_blur, self.voxel_sup_blur_frac, voxel_number_temp)
            """
            
            # testing
            #if self.voxel_sup_blur_frac is not None and self.voxel_sup_blur < 0.5 :
            #    self.voxel_sup_blur = 0.5
                
            if self.overlap == 'unit_cell' :
                print('\n\nupdating support with no unit_cell overlap')
                self.voxel_support = choose_N_highest_pixels(intensity, self.voxel_number, \
                                     support = self.support, syms = syms)
                #self.voxel_support = voxel_number_support_single_connected_region(intensity, self.voxel_number, init_sup=self.voxel_support)
            
            elif self.overlap == 'crystal' :
                print('\n\nupdating support with no crystal overlap')
                # try using the crystal mapping instead of the unit-cell mapping
                self.voxel_support = choose_N_highest_pixels(intensity, self.voxel_number, \
                                     support = self.support, syms = self.sym_ops.solid_to_crystal_real)
            elif self.overlap is None :
                print('\n\nupdating support')
                self.voxel_support = choose_N_highest_pixels(intensity, self.voxel_number, \
                                     support = self.support, syms = None)
            else :
                raise ValueError("overlap must be one of 'unit_cell', 'crystal' or None")

            # update sym_ops
            self.sym_ops.update_sup(self.voxel_support)
        
        
    def object(self, modes):
        self.O = self.sym_ops.av_solid_real(self.O, modes, allpix=True)
        return self.O.get()
        #return modes.get()[0]
        
    def Imap(self, modes):
        shape = modes.shape
        self.fftM3Dc(modes, modes)
        #cuda_stream.synchronize()
        self._make_I(np.int32(shape[1]*shape[2]*shape[3]), modes, self.DWg, self.BWg, self.Ig, block=(1024,1,1), grid=(32,1))
        return self.Ig
    
    def Psup(self, modes):
        # average 
        self.O = self.sym_ops.av_solid_real(self.O, modes)
        
        # reality
        self.O = (self.O + self.O.conj())/2.
        
        # broadcast
        modes = self.sym_ops.solid_syms_real(self.O, modes)
        return modes
    
    def Pmod(self, modes):
        # make x y
        #-----------------------------------------------
        self.fftM4Dc(modes, modes)
        modes /= np.sqrt(modes.shape[0])
        
        s = modes.shape
        self._make_xy(np.int32(s[0]), np.int32(s[1]*s[2]*s[3]), modes, 
                      self.xg, self.yg, block=(256,1,1), grid=(64,1))
        
        # project onto xp yp
        #-----------------------------------------------
        self._project_2D_Ellipse_arrays_cuda(np.int32(s[1]*s[2]*s[3]), self.xg, self.yg,
                                             self.Wxg, self.Wyg, self.I0g, self.maskg, 
                                             self.xpg, self.ypg,
                                             block=(256,1,1), grid=(64,1))
        
        # xp yp --> modes
        #-----------------------------------------------
        self._modes_xy(np.int32(s[0]), np.int32(s[1]*s[2]*s[3]), modes, 
                      self.xpg, self.ypg, self.yg, block=(256,1,1), grid=(64,1))
        
        
        modes *= np.sqrt(modes.shape[0])
        self.fftM4Dc(modes, modes, 1)
        
        return modes

    def Emod(self, modes):
        self.Ig   = self.Imap(modes)
        self.Ig   = ( cumath.sqrt(self.Ig) - self.ampg )**2
        eMod      = gpuarray.sum( self.Ig ).get()
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        #self.modes     = modes.get()
        out['O']        = self.object(modes)
        out['support']  = self.voxel_support
        out['diff']     = self.Imap(modes.copy()).get()
        out['mask']     = self.mask
        out['amp_diff'] = self.mask * (self.amp**2 - out['diff'])
        self.modes      = modes
        return out

    def l2norm(self, delta, array0):
        out = gpuarray.sum( (delta * delta.conj()).real ) / gpuarray.sum( (array0 * array0.conj()).real )
        return np.sqrt(out.get())

    def scans_cheshire(self, solid, scan_points=None, err = None):
        """
        err is deprecated
        """
        #scan_grid = [range(1), range(1), range(1)]
        # testing
        print('\n\nbluring sample...')
        import scipy.ndimage.filters
        from scipy.ndimage.filters import gaussian_filter
        O = gaussian_filter(solid.copy().real, 0.5, mode='wrap') + 0J

        scan_grid = [np.arange(-16, 16, 1) for i in range(3)]
        
        errors, shift = chesh_scan_P212121(self.amp**2, self.sym_ops.unitcell_size, O, 
                                           self.diffuse_weighting, self.unit_cell_weighting, 
                                           self.mask, scan_grid=scan_grid, check_finite=False)
        
        print('\n\nCheshire scan:', shift, 'shift', np.min(errors), 'error')
        #print('\n\nchecking the inverted object...')
        #errors2, shift2 = chesh_scan_P212121(self.amp**2, self.sym_ops.unitcell_size, solid[::-1,::-1,::-1], 
        #                                   self.diffuse_weighting, self.unit_cell_weighting, 
        #                                   self.mask, check_finite=False)
        #if np.min(errors2) < np.min(errors):
        #    print('choosing fliped object:')
        #    errors2 = errors
        #    sout  = solid[::-1, ::-1, ::-1]
        #    shift = shift2
        #    supout = self.voxel_support[::-1, ::-1, ::-1]
        #else :
        #    sout = solid
        #    supout = self.voxel_support
        sout   = solid
        supout = self.voxel_support
        
        print('\n\nCheshire scan:', shift, 'shift', np.min(errors), 'error')
        
        # now shift the sample and support
        sout = np.roll(sout, shift[0], 0)
        sout = np.roll(sout, shift[1], 1)
        sout = np.roll(sout, shift[2], 2)
        
        supout = np.roll(supout, shift[0], 0)
        supout = np.roll(supout, shift[1], 1)
        supout = np.roll(supout, shift[2], 2)
        
        self.voxel_support = supout.copy()
        
        # update sym_ops
        self.sym_ops.update_sup(self.voxel_support)
        
        # update the modes
        self.O     = gpuarray.to_gpu(np.ascontiguousarray(sout))
        self.modes = self.sym_ops.solid_syms_real(self.O, self.modes)
        
        # calculate errors
        info = {}
        info['eMod']      = [self.Emod(self.modes.copy())]
        info['error_map'] = errors
        info['eCon']      = [0.]
        info.update(self.finish(self.modes))
        return sout, info

def chesh_scan_P212121(diff, unit_cell, sin, D, B, mask, scan_grid=None, Bragg_mask=None, check_finite=False):
    import symmetry_operations
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
                         Mg.gpudata, Deltag.gpudata, maskg.gpudata, 
                         BWg.gpudata, Dg.gpudata, ampg.gpudata)
                
                # Emod
                eMod      = gpuarray.sum( Deltag ).get()
                eMod      = np.sqrt( eMod / I_norm )
                
                errors[ii, jj, kk] = eMod
        print(i, j, k, np.min(errors[ii]))
    
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

def choose_N_highest_pixels(array, N, tol = 1.0e-10, maxIters=1000, syms = None, support = None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)

    If support is not None then values outside the support
    are ignored. 
    """
    
    # no overlap constraint
    if syms is not None :
        # if array is not the maximum value
        # of the M symmetry related units 
        # then do not update 
        max_support = syms[0] > ((1-tol) * np.max(syms[1:], axis=0))
    else :
        max_support = np.ones(array.shape, dtype = np.bool)
    
    if support is not None and support is not 1 :
        sup = support
        a = array[(max_support > 0) * (support > 0)]
    else :
        sup = True
        a = array[(max_support > 0)]
    
    # search for the cutoff value
    s0 = array.max()
    s1 = array.min()
    
    failed = False
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = np.sum(a > s) - N
          
        if e == 0 :
            #print('e==0, exiting...')
            break
        
        if e < 0 :
            s0 = s
        else :
            s1 = s

        #print(s, s0, s1, e)
        if np.abs(s0 - s1) < tol and np.abs(e) > 0 :
            failed = True
            print('s0==s1, exiting...', s0, s1, np.abs(s0 - s1), tol)
            break
        
    S = (array > s) * max_support * sup
    
    # if failed is True then there are a lot of 
    # entries in a that equal s
    
    if False :
    #if failed :
        print('failed, sum(max_support), sum(S), voxels, pixels>0:',np.sum(max_support), np.sum(S), N, np.sum(array>0), len(a>0))
        # if S is less than the 
        # number of voxels then include 
        # some of the pixels where array == s
        count      = np.sum(S)
        ii, jj, kk = np.where((np.abs(array-s)<=tol) * (max_support * sup > 0))
        l          = N - count
        print(count, N, l, len(ii))
        if l > 0 :
            S[ii[:l], jj[:l], kk[:l]]    = True
        else :
            S[ii[:-l], jj[:-l], kk[:-l]] = False
    
    #print('number of pixels in support:', np.sum(S), i, s, e)
    return S
