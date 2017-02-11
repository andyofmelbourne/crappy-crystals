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
            N * lattice * exp
            
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
            O = args['solid_unit']
        else :
            print('initialising object with random numbers')
            O = np.random.random(I.shape).astype(c_dtype)
            print(O.shape)
        
        self.O = O
        Ohat   = np.fft.fftn(O)
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = 1
        if isValid('mask', args):
            print('setting mask...')
            self.mask = args['mask']
            print(np.sum(~self.mask), 'bad pixels')
            print(np.sum(self.mask), 'good pixels')
            print(self.mask.dtype, 'good pixels dtype')
        
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
        
        # diffuse and Bragg weightings
        #-----------------------------
        self.unit_cell_weighting = self.mask * args['Bragg_weighting']
        self.diffuse_weighting   = self.mask * args['diffuse_weighting']
        
        # make the reconstruction modes
        #------------------------------
        self.modes = np.zeros( (2 * self.sym_ops.no_solid_units,) + O.shape, O.dtype)
        
        # diffuse terms
        #self.modes[:self.modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(Ohat, apply_translation = False, syms = self.modes[:self.modes.shape[0]//2])
        self.modes[:self.modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(Ohat, apply_translation = False)
        
        # unit cell terms
        #self.modes[self.modes.shape[0]//2:] = self.sym_ops.solid_syms_Fourier(Ohat, apply_translation = True, syms = self.modes[self.modes.shape[0]//2:])
        self.modes[self.modes.shape[0]//2:] = self.sym_ops.solid_syms_Fourier(Ohat, apply_translation = True)
        
        # Ellipse axes
        #-----------------------------------------------
        # Here we have :
        # (x / e0)**2 + (y / e1)**2 = 1 , where
        # 
        # e0 = sqrt{ self.diffuse_weighting / I } and
        # 
        # e1 = sqrt{ self.unit_cell_weighting / I }
        # 
        # we need to know for which pixels:
        # e_0/1 -> inf
        #-----------------------------------------------
        
        # floating point tolerance for 1/x (log10)  
        tol = 15. #1.0e+15
        
        # check for numbers close to infinity in sqrt(I / self.diffuse_weighting)
        m = (self.diffuse_weighting <= 10.**(-tol)) + (I <= 10.**(-tol))
        m[~m] = (0.5 * (np.log10(I[~m]) - np.log10(self.diffuse_weighting[~m])) >= tol) 
        
        self.e0_inf   = m.copy()
        self.e0       = np.zeros_like(self.diffuse_weighting)

        # set e0 for pixels where I>0, diffuse>0, and I/diffuse < inf
        self.e0[~m]   = np.sqrt(I[~m]) / np.sqrt(self.diffuse_weighting[~m])
        
        # set e0 for pixels where I~0
        m = (I <= 10.**(-tol))
        self.e0[m]  = 0.
               
        # check for numbers close to infinity in sqrt(I / self.unit_cell_weighting)
        m     = (self.unit_cell_weighting <= 10.**(-tol)) + (I <= 10.**(-tol))
        m[~m] = 0.5 * (np.log10(I[~m]) - np.log10(self.unit_cell_weighting[~m])) >= tol
              
        self.e1_inf = m.copy()
        
        # set e1 for pixels where I>0, Bragg>0, and I/Bragg < inf
        self.e1     = np.zeros_like(self.unit_cell_weighting)
        self.e1[~m] = np.sqrt(I[~m]) / np.sqrt(self.sym_ops.no_solid_units * self.unit_cell_weighting[~m])
        
        # set e1 for pixels where I~0
        m = (I <= 10.**(-tol))
        self.e1[m] = 0.
        
        print('number of good pixels for elliptical projections: e0, e1, both', np.sum(~self.e0_inf), np.sum(~self.e1_inf), np.sum(~self.e1_inf * ~self.e0_inf))
        
        self.iters = 0
        
        # check that self.Imap == I * (x/e_0)**2 + (y/e_1)**2
        # or that (x/e_0)**2 + (y/e_1)**2 = 1
        self.e0_inf = self.e0_inf.astype(np.uint8)
        self.e1_inf = self.e1_inf.astype(np.uint8)
        print('eMod(modes0):', self.Emod(self.modes))

         
    def object(self, modes):
        out = np.fft.ifftn(modes[0])
        return out
    
    def Imap(self, modes):
        U  = np.sum(modes[modes.shape[0]//2 :], axis=0)
        D  = modes[: modes.shape[0]//2]
        
        I  = self.diffuse_weighting   * np.sum( (D * D.conj()).real, axis=0)
        I += self.unit_cell_weighting * (U * U.conj()).real
        return I
    
    def Psup(self, modes):
        #return modes.copy()
        out = np.empty_like(modes)
        
        # diffuse terms: unflip the modes
        out[: modes.shape[0]//2] = \
                self.sym_ops.unflip_modes_Fourier(modes[: modes.shape[0]//2], apply_translation = False)
        
        # unit_cell terms: unflip the modes
        out[modes.shape[0]//2 :] = \
                self.sym_ops.unflip_modes_Fourier(modes[modes.shape[0]//2 :], apply_translation = True)
        
        # average 
        out = np.mean(out, axis=0)
        
        # propagate
        out = np.fft.ifftn(out)

        # finite support
        if self.voxel_number :
            if self.overlap == 'unit_cell' :
                self.voxel_support = choose_N_highest_pixels( (out * out.conj()).real, self.voxel_number, \
                                     support = self.support, mapper = self.sym_ops.solid_syms_real)
            elif self.overlap == 'crystal' :
                # try using the crystal mapping instead of the unit-cell mapping
                self.voxel_support = choose_N_highest_pixels( (out * out.conj()).real.astype(np.float32), self.voxel_number, \
                                     support = self.support, mapper = self.sym_ops.solid_to_crystal_real)
            elif self.overlap is None :
                # try using the crystal mapping instead of the unit-cell mapping
                self.voxel_support = choose_N_highest_pixels( (out * out.conj()).real.astype(np.float32), self.voxel_number, \
                                     support = self.support, mapper = None)
            else :
                raise ValueError("overlap must be one of 'unit_cell', 'crystal' or None")
        
        out *= self.voxel_support

        # reality
        out.imag = 0
        
        # store the latest guess for the object
        self.O = out.copy()
        
        # propagate
        out = np.fft.fftn(out)
        
        # broadcast
        modes_out = np.empty_like(self.modes)
        modes_out[: modes_out.shape[0]//2] = self.sym_ops.solid_syms_Fourier(out, apply_translation=False, syms=modes_out[: modes_out.shape[0]//2])
        modes_out[modes_out.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(out, apply_translation=True,  syms=modes_out[modes_out.shape[0]//2 :])

        self.iters += 1
        
        #print(' sum | sqrt(I) - sqrt(Imap) | : ', self.Emod(modes_out))
        return modes_out

    def Pmod(self, modes):
        U  = modes[modes.shape[0]//2 :]
        D  = modes[: modes.shape[0]//2]
        
        # make x
        #-----------------------------------------------
        x = np.sqrt(np.sum( (D * D.conj()).real, axis=0)).ravel()
        
        # make y
        #-----------------------------------------------
        # rotate the unit cell modes
        # eg. for N = 2 : Ut = np.array([[1., 1.], [-1., 1.]], dtype=np.float64) / np.sqrt(2.)
        # Ut = make_unitary_transform(modes.shape[0]//2)
        # u = np.dot(Ut, U.reshape((U.shape[0], -1)))
        
        # actually this is just an fft!
        # just have to make sure that it is norm preserving
        #u = (np.fft.fftn(U, axes=(0,))/np.sqrt(U.shape[0])).reshape((U.shape[0], -1))
        u = (np.fft.fftn(U, axes=(0,))/np.sqrt(U.shape[0]))[0].ravel()
        
        y = np.abs(u) # np.abs(np.sum(U, axis=0) / np.sqrt(U.shape[0]))
        
        print('y sum:', np.sum(y))
        # look for forbidden reflections where e1_inf != 1
        forbidden   = (y < 1000*self.alpha) * (self.e1_inf != 1).ravel()
        
        # project onto xp yp
        #-----------------------------------------------
        xp, yp = project_2D_Ellipse_arrays_cython(self.e0.ravel(), self.e1.ravel(), 
                                                  x, y, 
                                                  self.e0_inf.ravel(), self.e1_inf.ravel())
        
        print('yp sum:', np.sum(yp))
        # xp yp --> modes
        #-----------------------------------------------
        # d *= xp / x
        #############
        out = modes.copy()
        out[: modes.shape[0]//2] *= (xp / (x + self.alpha)).reshape(D[0].shape)
        print('D sum:', np.sum(np.abs(out[: modes.shape[0]//2])**2))
        
        # u   *= yp / y
        # U[0] = u
        # U    = ifft(U)
        ################
        # for forbidden reflections be more careful 
        angle = np.angle(u[forbidden])
        up            = np.empty_like(u)
        up[forbidden] = yp[forbidden] * np.exp(1J * angle)
        print('up sum:', np.sum(np.abs(up)**2))
        
        # for everything else just scale (this is much faster)
        up[~forbidden] = u[~forbidden] * yp[~forbidden] / (y[~forbidden] + self.alpha)
        
        print('up sum:', np.sum(np.abs(up)**2))
        # un-rotate
        #out[modes.shape[0]//2 :] = np.sqrt(U.shape[0])*np.fft.ifftn(u, axes=(0,)).reshape(D.shape)
        out[modes.shape[0]//2 :] += (up - u).reshape(D[0].shape) / np.sqrt(U.shape[0])
        
        # store the latest eMod, but just use one mode to speed this up
        delta     = modes[4] - out[4]
        self.eMod = self.l2norm(delta, out[4])
        # check
        #print(' sum | sqrt(I) - sqrt(Imap) | : ', self.Emod(out))
        #print('Pmod -->  sum | modes - out | : ', np.sum( np.abs(modes - out) ))
        return out

    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = np.sum( self.mask * ( np.sqrt(M) - self.amp )**2 )
        eMod      = np.sqrt( eMod / self.I_norm )
        
        #if eMod < 1.0e-1 :
        #    import h5py 
        #    f = h5py.File('test')
        #    f['amp'] = self.amp
        #    f['amp_forward'] = np.sqrt(M)
        #    f.close()
        #    sys.exit()
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

    def scans_cheshire_old(self, solid, step=4):
        """
        scan the solid unit through the cheshire cell 
        until the best agreement with the data is found.
        """
        
        s = phasing_3d.utils.merge.centre(solid)
        I, J, K = self.sym_ops.unitcell_size
        I //= 2
        J //= 2
        K //= 2
        modes = np.empty_like(self.modes)
        errors = np.zeros((I, J, K), dtype=np.float)
        errors.fill(np.inf)
        for i in range(0, I, step):
            for j in range(0, J, step):
                for k in range(0, K, step):
                    # shift
                    s1 = phasing_3d.utils.merge.multiroll(s, [i,j,k])
                    
                    # propagate
                    s1 = np.fft.fftn(s1)
                    
                    # broadcast
                    modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=False)
                    modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True)
                    
                    errors[i, j, k] = self.Emod(modes)
                    print(i, j, k, errors[i,j,k])
        
        l = np.argmin(errors)
        i, j, k = np.unravel_index(l, errors.shape)
        print('lowest error at: i, j, k, err', i, j, k, errors[i,j,k])
          
        # shift
        s1 = phasing_3d.utils.merge.multiroll(s, [i,j,k])
        
        # propagate
        s1 = np.fft.fftn(s1)
        
        # broadcast
        modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=False)
        modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True)
        
        s1 = phasing_3d.utils.merge.multiroll(s, [i,j,k])
        
        info = {}
        info['eMod'] = [errors[i, j, k]]
        info['error_map'] = errors[0:I:step, 0:J:step, 0:K:step]
        info['eCon'] = [self.l2norm(self.modes - modes, modes)]
        info.update(self.finish(modes))
        return s1, info

    def scans_cheshire_newer_old(self, solid, steps=[4,4,4], unit_cell=False, err = 'Emod'):
        """
        scan the solid unit through the cheshire cell 
        until the best agreement with the data is found.
        """
        if err == 'Emod' :
            err = self.Emod 

        I, J, K = self.sym_ops.unitcell_size
        if unit_cell is False :
            I //= 2
            J //= 2
            K //= 2
        modes = np.empty_like(self.modes)
        errors = np.zeros((I, J, K), dtype=np.float)
        errors.fill(np.inf)
        # propagate
        s = np.fft.fftn(solid)
        
        ii = np.fft.fftfreq(s.shape[0]) 
        jj = np.fft.fftfreq(s.shape[1])
        kk = np.fft.fftfreq(s.shape[2])
        for i in range(0, I, steps[0]):
            T0 = np.exp(- 2J * np.pi * i * ii)
            for j in range(0, J, steps[1]):
                T1 = np.exp(- 2J * np.pi * j * jj)
                for k in range(0, K, steps[2]):
                    T2 = np.exp(- 2J * np.pi * k * kk)
                     
                    phase_ramp = reduce(np.multiply.outer, [T0, T1, T2])
                    s1         = s * phase_ramp
                     
                    # broadcast
                    modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=False, syms = modes[: modes.shape[0]//2])
                    modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True, syms = modes[modes.shape[0]//2 :])
                    
                    errors[i, j, k] = err(modes)
                    print(i, j, k, errors[i,j,k])
        
        l = np.argmin(errors)
        i, j, k = np.unravel_index(l, errors.shape)
        print('lowest error at: i, j, k, err', i, j, k, errors[i,j,k])
          
        # shift
        T0 = np.exp(- 2J * np.pi * i * ii)
        T1 = np.exp(- 2J * np.pi * j * jj)
        T2 = np.exp(- 2J * np.pi * k * kk)
        phase_ramp = reduce(np.multiply.outer, [T0, T1, T2])
        s1         = s * phase_ramp
        
        # broadcast
        modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=False)
        modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True)
        
        s1 = np.fft.ifftn(s1)
        
        info = {}
        info['eMod'] = [errors[i, j, k]]
        info['error_map'] = errors[0:I:steps[0], 0:J:steps[1], 0:K:steps[2]]
        info['eCon'] = [self.l2norm(self.modes - modes, modes)]
        info.update(self.finish(modes))
        return s1, info
    
    def scans_cheshire(self, solid, steps=[1,1,1], unit_cell=False, err = 'Emod'):
        """
        scan the solid unit through the cheshire cell 
        until the best agreement with the data is found.
        """
        if err == 'Emod' :
            err = self.Emod 

        I, J, K = self.sym_ops.unitcell_size
        if unit_cell is False :
            I //= 2
            J //= 2
            K //= 2
        
        errors = np.zeros((I, J, K), dtype=np.float)
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
        e0_inf = self.e0_inf[Bragg_mask] == 0 
        e1_inf = self.e1_inf[Bragg_mask] == 0 
        modes  = np.empty((self.modes.shape[0],)+s[Bragg_mask].shape, dtype=self.modes.dtype)
        amp    = self.amp[Bragg_mask] 
        if self.mask is not 1 :
            mask = self.mask[Bragg_mask] 
        else :
            mask = 1
        I_norm = np.sum(mask*amp**2)
        diffuse_weighting    = self.diffuse_weighting[Bragg_mask]
        unit_cell_weighting  = self.unit_cell_weighting[Bragg_mask]
        
        for i in range(0, I, steps[0]):
            for j in range(0, J, steps[1]):
                for k in range(0, K, steps[2]):
                    phase_ramp = np.exp(- 2J * np.pi * (i * qi + j * qj + k * qk))
                     
                    s1[Bragg_mask] = sBragg * phase_ramp
                     
                    # broadcast
                    modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier_masked(s1, ii, jj, kk, apply_translation=False, syms = modes[: modes.shape[0]//2])
                    modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier_masked(s1, ii, jj, kk, apply_translation=True,  syms = modes[modes.shape[0]//2 :])
                    
                    # I map
                    U  = np.sum(modes[modes.shape[0]//2 :], axis=0)
                    D  = modes[: modes.shape[0]//2]
                     
                    I  = diffuse_weighting   * np.sum( (D * D.conj()).real, axis=0)
                    I += unit_cell_weighting * (U * U.conj()).real
                    
                    # Emod
                    mask      = mask * e0_inf * e1_inf
                    eMod      = np.sum( mask * ( np.sqrt(I) - amp )**2 )
                    eMod      = np.sqrt( eMod / I_norm )

                    errors[i, j, k] = eMod
        
        l = np.argmin(errors)
        i, j, k = np.unravel_index(l, errors.shape)
        print('lowest error at: i, j, k, err', i, j, k, errors[i,j,k])
          
        # shift
        qi = np.fft.fftfreq(s.shape[0]) 
        qj = np.fft.fftfreq(s.shape[1])
        qk = np.fft.fftfreq(s.shape[2])
        T0 = np.exp(- 2J * np.pi * i * qi)
        T1 = np.exp(- 2J * np.pi * j * qj)
        T2 = np.exp(- 2J * np.pi * k * qk)
        phase_ramp = reduce(np.multiply.outer, [T0, T1, T2])
        s1         = s * phase_ramp
        
        # broadcast
        modes  = np.empty(self.modes.shape, dtype=self.modes.dtype)
        modes[: modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=False)
        modes[modes.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(s1, apply_translation=True)
        
        s1 = np.fft.ifftn(s1)
        
        info = {}
        info['eMod'] = [errors[i, j, k]]
        info['error_map'] = errors
        info['eCon'] = [self.l2norm(self.modes - modes, modes)]
        info.update(self.finish(modes))
        return s1, info

def make_unitary_transform(N):
    U = np.zeros((N, N), dtype=np.float)
    U[0, :] = 1  / np.sqrt(N)
    U[1:, 0] = -1 / np.sqrt(N)
    for n in range(1, N):
        for m in range(1, N):
            if n == m :
                U[n, m] = (N * (N - 2) + np.sqrt(N)) / ( (N - 1)*N )
            else :
                U[n, m] = (-N + np.sqrt(N)) / ( (N - 1)*N )

    # check:
    # assert np.allclose(np.dot(U.T, U), np.identity(N))
    return U


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
        a = array[max_support]
    else :
        max_support = np.ones(array.shape, dtype = np.bool)
        a = array
    
    if support is not None and support is not 1 :
        a = array[support > 0]
    else :
        support = 1
    
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
            print('s0==s1, exiting...')
            break
        
    S = (array > s) * max_support * support
    
    # if failed is True then there are a lot of 
    # entries in a that equal s
    if failed :
        print('failed, sum(max_support), sum(S), voxels:',np.sum(max_support), np.sum(S), N)
        # if S is less than the 
        # number of voxels then include 
        # some of the pixels where array == s
        count      = np.sum(S)
        ii, jj, kk = np.where(np.abs(array-s)<=tol)
        l          = N - count
        print(count, N, l, len(ii))
        if l > 0 :
            S[ii[:l], jj[:l], kk[:l]]    = True
        else :
            S[ii[:-l], jj[:-l], kk[:-l]] = False
    
    #print('number of pixels in support:', np.sum(S), i, s, e)
    return S
