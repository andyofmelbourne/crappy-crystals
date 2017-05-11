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

import symmetry_operations 
import padding
import add_noise_3d
import io_utils

import pyximport; pyximport.install()
from ellipse_2D_cython import project_2D_Ellipse_arrays_cython
from ellipse_2D_cython_new import project_2D_Ellipse_arrays_cython_test

import phasing_3d
from phasing_3d.src.mappers import Modes
from phasing_3d.src.mappers import isValid


def get_sym_ops(space_group, unit_cell, det_shape):
    if space_group == 'P1':
        print('\ncrystal space group: P1')
        sym_ops = \
            symmetry_operations.P1(unit_cell, det_shape)

    elif space_group == 'P212121':
        print('\ncrystal space group: P212121')
        sym_ops = \
            symmetry_operations.P212121(unit_cell, det_shape)
    
    elif space_group == 'Ptest':
        print('\ncrystal space group: Ptest')
        sym_ops = \
            symmetry_operations.Ptest(unit_cell, det_shape)

    return sym_ops


class Mapper_ellipse():

    def __init__(self, I, **args):
        """
        A class for performing mapping operations on crappy crystal modes.

        This class profides an object that can be used by 3D-phasing, which
        requires the following methods:
            I     = mapper.Imap(modes)   # mapping the modes to the intensity
            modes = mapper.Pmod(modes)   # applying the data projection to the modes
            modes = mapper.Psup(modes)   # applying the support projection to the modes
            O     = mapper.object(modes) # the main object of interest
            dict  = mapper.finish(modes) # add any additional output to the info dict
        
        Parameters
        ----------
        I : numpy.ndarray, float
            The 3D diffraction volume
        
        Keyword Arguments
        -----------------
        'Bragg_weighting' : numpy.ndarray, float
            lattice * exp
            
        'diffuse_weighting' : numpy.ndarray, float
            (1 - exp)
        
        solid_unit : numpy.ndarray, optional, default (None)
            The solid_unit of the crystal. If None then it is initialised
            with random numbers.
        
        mask : numpy.ndarray, optional, default (None)
            bad pixel mask for the diffraction volume. If None then every 
            pixel is used. Bad pixels = False
        
        support : numpy.ndarray, optional, default (None)
            A fixed support volume, if support[i] = 0 then the object is
            not at pixel i  
        
        voxels : integer, optional, default (None)
            The number of pixels that the solid_unit can occupy.
        
        overlap : ('unit_cell', 'crystal', None), optional, default (None)
            Prevent overlap of the solid unit when applying the support.  

        sym : object, optional, default (None)
            A crystal symmetry operator, if None then this object is created 
            with 'unit_cell' and 'space_group' (see below).
        
        unit_cell : sequence of length 3, int
            The pixel dimensions of the unit-cell
        
        space_group : string, optional, default ('P1')
        
        alpha : float, optional, default (1.0e-10)
            floating point offset to prevent divide by zeros: a / (b + alpha)
        
        dtype : np.dtype, optional, default (np.float64)
            the complex data type is inferred from this
        """
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
        
        self.O = O
        Ohat   = np.fft.fftn(O)
        
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
        if isValid('mask', args):
            print('setting mask...')
            self.mask = args['mask']
            
        # mask pixels with low weighting factors
        #tol = 1.0e-9
        #self.mask[(self.diffuse_weighting < tol) * (self.unit_cell_weighting < tol)] = False
        #self.mask[I<tol] = False
        #print(np.sum(~self.mask), 'bad pixels')
        #print(np.sum(self.mask), 'good pixels')
        #print(self.mask.dtype, 'good pixels dtype')
        
        self.I_norm = (self.mask * I).sum()
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']

        self.I_norm = (self.mask * I).sum()
        self.amp    = np.sqrt(I.astype(dtype))

        # define the support projection
        #-----------------------------------------------
        if isValid('support', args) :
            self.support = args['support']
        else :
            self.support = 1
        
        if isValid('voxels', args) :
            self.voxel_number  = args['voxels']
            self.voxel_support = np.ones(O.shape, dtype=np.bool)
        else :
            self.voxel_number  = False
            self.voxel_support = self.support.copy()
        
        if isValid('overlap', args) :
            self.overlap = args['overlap']
        else :
            self.overlap = None
        
        # make the crystal symmetry operator
        #-----------------------------------
        if isValid('sym', args):
            self.sym_ops = args['sym'] 
        else :
            self.sym_ops = get_sym_ops(args['space_group'], args['unit_cell'], O.shape)
        
        # make the reconstruction modes
        #------------------------------
        self.modes = np.zeros( (self.sym_ops.no_solid_units,) + O.shape, O.dtype)
        
        self.modes = self.sym_ops.solid_syms_Fourier(Ohat, apply_translation = True)
        
        # precalculate the ellipse projection arguments
        #----------------------------------------------
        self.Wx         = (self.diffuse_weighting + self.sym_ops.no_solid_units * self.unit_cell_weighting).ravel()
        self.Wy         = self.diffuse_weighting.ravel()
        self.I_ravel    = I.astype(dtype).ravel()
        self.mask_ravel = self.mask.astype(np.uint8).ravel()
        
        # check that self.Imap == I * (x/e_0)**2 + (y/e_1)**2
        # or that (x/e_0)**2 + (y/e_1)**2 = 1
        self.iters = 0
        
        print('eMod(modes0):', self.Emod(self.modes))

         
    def object(self, modes):
        out = np.fft.ifftn(modes[0])
        return out
    
    def Imap(self, modes):
        U  = np.sum(modes, axis=0)
        I  = self.diffuse_weighting   * np.sum( (modes * modes.conj()).real, axis=0)
        I += self.unit_cell_weighting * (U * U.conj()).real
        return I
    
    def Psup(self, modes):
        out = modes.copy()
        
        # unit_cell terms: unflip the modes
        out = self.sym_ops.unflip_modes_Fourier(out, apply_translation = True, inplace=True)
        
        # average 
        out_solid = np.mean(out, axis=0)
        
        # propagate
        out_solid = np.fft.ifftn(out_solid)
        
        # reality
        out_solid.imag = 0
        
        # finite support
        if self.voxel_number :
            #print('\n\nVoxel number support')
            if self.overlap == 'unit_cell' :
                self.voxel_support = choose_N_highest_pixels( (out_solid * out_solid.conj()).real.astype(np.float32), self.voxel_number, \
                                     support = self.support, mapper = self.sym_ops.solid_syms_real)
            elif self.overlap == 'crystal' :
                # try using the crystal mapping instead of the unit-cell mapping
                self.voxel_support = choose_N_highest_pixels( (out_solid * out_solid.conj()).real.astype(np.float32), self.voxel_number, \
                                     support = self.support, mapper = self.sym_ops.solid_to_crystal_real)
            elif self.overlap is None :
                self.voxel_support = choose_N_highest_pixels( (out_solid * out_solid.conj()).real.astype(np.float32), self.voxel_number, \
                                     support = self.support, mapper = None)
            else :
                raise ValueError("overlap must be one of 'unit_cell', 'crystal' or None")
        
        out_solid *= self.voxel_support
        
        # store the latest guess for the object
        self.O = out_solid.copy()
        
        # propagate
        out_solid = np.fft.fftn(out_solid)
        
        # broadcast
        out = self.sym_ops.solid_syms_Fourier(out_solid, apply_translation=True,  syms=out)
        
        self.iters += 1
        
        return out

    def Pmod(self, modes):
        u = np.fft.fftn(modes, axes=(0,)).reshape((modes.shape[0], -1)) / np.sqrt(modes.shape[0])
        
        # make x
        #-----------------------------------------------
        x = np.sqrt((u[0] * u[0].conj()).real).ravel()
        
        # make y
        #-----------------------------------------------
        if u.shape[0] > 1 :
            y = np.sqrt(np.sum( (u[1:] * u[1:].conj()).real, axis=0))
        else :
            y = np.zeros_like(x)
        
        # project onto xp yp
        #-----------------------------------------------
        xp, yp = project_2D_Ellipse_arrays_cython_test(x, y,
                                          self.Wx,
                                          self.Wy,
                                          self.I_ravel,
                                          self.mask_ravel)
        
        # xp yp --> modes
        #-----------------------------------------------
        #u[0]  = u[0]  * xp / (x + self.alpha)
        angle = np.angle(u[0])
        u[0]  = xp * np.exp(1J*angle)
        
        u[1:] = u[1:] * yp / (y + self.alpha)
        
        # un-rotate
        u = np.fft.ifftn(u, axes=(0,)) * np.sqrt(modes.shape[0])
        
        out = u.reshape(modes.shape)
        return out

    def Emod(self, modes):
        M         = self.Imap(modes)
        M         = self.mask * ( np.sqrt(M) - self.amp )**2
        eMod      = np.sum( M )
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def Esup(self, modes):
        M         = self.Psup(modes)
        M        -= modes
        eSup      = np.sum( (M * M.conj() ).real ) 
        eSup      = np.sqrt( eSup / np.sum( (modes * modes.conj()).real ))
        return eSup

    def finish(self, modes):
        out = {}
        self.modes     = modes
        self.O         = self.object(modes)
        out['support'] = self.voxel_support
        out['diff']    = self.Imap(modes)
        out['mask']    = self.mask
        out['amp_diff'] = self.mask * (self.amp**2 - out['diff'])
        return out

    def l2norm(self, delta, array0):
        num = 0
        den = 0
        #print('l2norm --> np.sum(|delta|**2)', np.sum(np.abs(delta)**2))
        #print('l2norm --> np.sum(|array0|**2)', np.sum(np.abs(array0)**2))
        for i in range(delta.shape[0]):
            num += np.sum( (delta[i] * delta[i].conj()).real ) 
            den += np.sum( (array0[i] * array0[i].conj()).real ) 
        return np.sqrt(num / den)

    def scans_cheshire(self, solid, scan_points=None, err = 'Emod'):
        """
        scan the solid unit through the cheshire cell 
        until the best agreement with the data is found.
        """
        if err == 'Emod' :
            err = self.Emod 
        
        if scan_points is not None :
            #I, J, K = self.sym_ops.Cheshire_cell
            I = scan_points[0]
            J = scan_points[1]
            K = scan_points[2]
        else :
            I = range(self.sym_ops.unitcell_size[0])
            J = range(self.sym_ops.unitcell_size[1])
            K = range(self.sym_ops.unitcell_size[2])
        
        errors = np.zeros((len(I), len(J), len(K)), dtype=np.float)
        errors.fill(np.inf)
        
        # only evaluate the error on Bragg peaks that are strong 
        Bragg_mask = self.unit_cell_weighting > 1.0e-1 * self.unit_cell_weighting[0,0,0]
        
        # symmetrise it so that we have all pairs in the point group
        Bragg_mask = self.sym_ops.solid_syms_Fourier(Bragg_mask, apply_translation=False)
        Bragg_mask = np.sum(Bragg_mask, axis=0)>0
        
        # propagate
        s  = np.fft.fftn(solid)
        s1 = np.zeros_like(s)
        
        qi = np.fft.fftfreq(s.shape[0]) 
        qj = np.fft.fftfreq(s.shape[1])
        qk = np.fft.fftfreq(s.shape[2])
        qi, qj, qk = np.meshgrid(qi, qj, qk, indexing='ij')
        qi = qi[Bragg_mask]
        qj = qj[Bragg_mask]
        qk = qk[Bragg_mask]
        
        ii = np.fft.fftfreq(s.shape[0], 1./s.shape[0]).astype(np.int)
        jj = np.fft.fftfreq(s.shape[1], 1./s.shape[1]).astype(np.int)
        kk = np.fft.fftfreq(s.shape[2], 1./s.shape[2]).astype(np.int)
        ii, jj, kk = np.meshgrid(ii, jj, kk, indexing='ij')
        ii = ii[Bragg_mask]
        jj = jj[Bragg_mask]
        kk = kk[Bragg_mask]
        
        sBragg = s[Bragg_mask]
        modes  = np.empty((self.modes.shape[0],)+s[Bragg_mask].shape, dtype=self.modes.dtype)
        amp    = self.amp[Bragg_mask] 
        if self.mask is not 1 :
            mask = self.mask[Bragg_mask] 
        else :
            mask = 1
        I_norm = np.sum(mask*amp**2)
        diffuse_weighting    = self.diffuse_weighting[Bragg_mask]
        unit_cell_weighting  = self.unit_cell_weighting[Bragg_mask]
        
        for i in I:
            for j in J:
                for k in K:
                    phase_ramp = np.exp(- 2J * np.pi * (i * qi + j * qj + k * qk))
                     
                    s1[Bragg_mask] = sBragg * phase_ramp
                     
                    # broadcast
                    modes = self.sym_ops.solid_syms_Fourier_masked(s1, ii, jj, kk, apply_translation=True,  syms = modes)
                    
                    # I map
                    U  = np.sum(modes, axis=0)
                     
                    diff  = diffuse_weighting   * np.sum( (modes * modes.conj()).real, axis=0)
                    diff += unit_cell_weighting * (U * U.conj()).real
                    
                    # Emod
                    mask      = mask 
                    eMod      = np.sum( mask * ( np.sqrt(diff) - amp )**2 )
                    eMod      = np.sqrt( eMod / I_norm )

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
        
        # broadcast
        modes = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True)
        
        s1 = np.fft.ifftn(s1)
        
        info = {}
        info['eMod'] = [self.Emod(modes)]
        info['error_map'] = errors
        info['eCon'] = [self.l2norm(self.modes - modes, modes)]
        info.update(self.finish(modes))
        return s1, info

def choose_N_highest_pixels(array, N, tol = 1.0e-10, maxIters=1000, mapper = None, support = None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)

    If support is not None then values outside the support
    are ignored. 
    """
    
    # no overlap constraint
    if mapper is not None :
        syms = mapper(array)
        # if array is not the maximum value
        # of the M symmetry related units 
        # then do not update 
        max_support = syms[0] == np.max(syms, axis=0) 
    else :
        max_support = np.ones(array.shape, dtype = np.bool)
    
    if support is not None and support is not 1 :
        sup = support
        a = array[(max_support > 0) * (support > 0)]
    else :
        sup = 1
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
            print('s0==s1, exiting...', s0, s1)
            break
        
    S = (array > s) * max_support * sup
    
    # if failed is True then there are a lot of 
    # entries in a that equal s
    if failed :
        print('failed, sum(max_support), sum(S), voxels, pixels>0:',np.sum(max_support), np.sum(S), N, np.sum(array>0), len(a>0))
        # if S is less than the 
        # number of voxels then include 
        # some of the pixels where array == s
        count      = np.sum(S)
        ii, jj, kk = np.where((np.abs(array-s)<=tol) * (max_support * support > 0))
        l          = N - count
        print(count, N, l, len(ii))
        if l > 0 :
            S[ii[:l], jj[:l], kk[:l]]    = True
        else :
            S[ii[:-l], jj[:-l], kk[:-l]] = False
    
    #print('number of pixels in support:', np.sum(S), i, s, e)
    return S
