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
            O = args['solid_unit'].astype(c_dtype)
        else :
            print('initialising object with random numbers')
            O = np.random.random(I.shape).astype(c_dtype)
            print(O.shape)
        
        self.O = O
        Ohat   = np.fft.fftn(O)
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = np.ones(I.shape, dtype=np.bool)
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
        
        tol = 1.0e-10
        
        self.e0     = np.zeros_like(self.diffuse_weighting)
        self.e0_inf = np.zeros(self.diffuse_weighting.shape, dtype = np.bool)
        
        self.e1     = np.zeros_like(self.diffuse_weighting)
        self.e1_inf = np.zeros(self.diffuse_weighting.shape, dtype = np.bool)
        
        # look for special cases
        D0 = (self.diffuse_weighting   <= tol)
        B0 = (self.unit_cell_weighting <= tol)
        I0 = (I                        <= tol)
        m  = self.mask > 0 

        # 'normal, on Bragg, non-masked' D>0, B>0, I>0, m != 0
        # and
        # 'noisey, on Bragg, non-masked' D>0, B>0, I=0, m != 0
        i = (~D0)*(~B0)*m
        print(i.shape, i.dtype, np.sum(i))
        self.e0[i] = np.sqrt(I[i]) / np.sqrt(self.diffuse_weighting[i])
        self.e1[i] = np.sqrt(I[i]) / np.sqrt(self.sym_ops.no_solid_units * self.unit_cell_weighting[i])
        
        self.e0_inf[i] = False
        self.e1_inf[i] = False
        print(np.sum(i), 'normal / noisey, diffuse, on Bragg, non-masked pixels')
        
        # 'normal, off Bragg, non-masked' D>0, B=0, I>0, m != 0
        # and
        # 'noisey, off Bragg, non-masked' D>0, B=0, I=0, m != 0
        i = (~D0)*(B0)*m
        self.e0[i] = np.sqrt(I[i]) / np.sqrt(self.diffuse_weighting[i])
        self.e1[i] = np.inf
        
        self.e0_inf[i] = False
        self.e1_inf[i] = True
        print(np.sum(i), 'normal / noisey, diffuse, off Bragg, non-masked pixels')
        
        
        # 'normal, no diffuse, on Bragg, non-masked' D=0, B>0, I>0, m != 0
        # and
        # 'noisey, no diffuse, on Bragg, non-masked' D=0, B>0, I=0, m != 0
        i = (D0)*(~B0)*m
        self.e0[i] = np.inf
        self.e1[i] = np.sqrt(I[i]) / np.sqrt(self.sym_ops.no_solid_units * self.unit_cell_weighting[i])
        
        self.e0_inf[i] = True
        self.e1_inf[i] = False
        print(np.sum(i), 'normal / noisey, no diffuse, on Bragg, non-masked pixels')

        # 'masked' D>=0, B>=0, I>=0, m = 0
        if m is not 1 and m is not True :
            i = ~m
            self.e0[i] = np.inf
            self.e1[i] = np.inf
             
            self.e0_inf[i] = True
            self.e1_inf[i] = True
            print(np.sum(i), 'masked pixels')

        # test: try masking pixels for which D>0 or B>0 and I==0 
        i = (D0 + B0)*(I0)*m
        self.e0[i] = np.inf
        self.e1[i] = np.inf
         
        self.e0_inf[i] = True
        self.e1_inf[i] = True
        self.mask[i] = False

        # 'everything zero pixels' D=0, B=0, I=0 non-masked
        i = (D0)*(B0)*(I0)*m
        self.e0[i] = 0.
        self.e1[i] = 0.
         
        self.e0_inf[i] = True
        self.e1_inf[i] = True
        print(np.sum(i), 'everything zero pixels')
        
        # 'weired pixels' D=0, B=0, I>=0 non-masked
        # for now raise an exception if there are any of these
        i = (D0)*(B0)*(~I0)*m
        if np.sum(i) > 0 :
            raise ValueError('Error diffuse and unit-cell weighting are zero on a pixel with positive intensity')

        self.e0_inf = self.e0_inf.astype(np.uint8)
        self.e1_inf = self.e1_inf.astype(np.uint8)
        self.I0     = I0.astype(np.uint8)
        """
        # floating point tolerance for 1/x (log10)  
        tol = 10. #1.0e+15
        
        # check for numbers close to infinity in sqrt(I / self.diffuse_weighting)
        #m = (self.diffuse_weighting <= 10.**(-tol)) + (I <= 10.**(-tol))
        #m[~m] = (0.5 * (np.log10(I[~m]) - np.log10(self.diffuse_weighting[~m])) >= tol) 
        self.I0 = (I <= 10.**(-tol)) 
        
        # e0 is infinite when the weighting factor is really small
        self.e0_inf   = np.zeros(self.diffuse_weighting.shape, dtype = np.bool)
        self.e0_inf[ (self.diffuse_weighting <= 10.**(-tol)) ] = True
        
        # set e0 to >0 for pixels where I>0 and diffuse>0
        self.e0    = np.zeros_like(self.diffuse_weighting)
        m          = (I>= 10.**(-tol)) * ~self.e0_inf
        self.e0[m] = np.sqrt(I[m]) / np.sqrt(self.diffuse_weighting[m])
        
        # set e0 for pixels where I~0
        m = (I <= 10.**(-tol))
        self.e0[m]  = 0.
               
        # e1 is infinite when the weighting factor is really small
        self.e1_inf   = np.zeros(self.unit_cell_weighting.shape, dtype = np.bool)
        self.e1_inf[ (self.unit_cell_weighting <= 10.**(-tol)) ] = True
        
        # set e1 to >0 for pixels where I>0 and Bragg>0
        self.e1    = np.zeros_like(self.unit_cell_weighting)
        m          = (I>= 10.**(-tol)) * ~self.e1_inf
        self.e1[m] = np.sqrt(I[m]) / np.sqrt(self.sym_ops.no_solid_units * self.unit_cell_weighting[m])
        
        # set e0 for pixels where I~0
        m = (I <= 10.**(-tol))
        self.e1[m]  = 0.
        
        print('number of good pixels for elliptical projections: e0, e1, both', np.sum(~self.e0_inf), np.sum(~self.e1_inf), np.sum(~self.e1_inf * ~self.e0_inf))
        """
        
        self.iters = 0
        
        # check that self.Imap == I * (x/e_0)**2 + (y/e_1)**2
        # or that (x/e_0)**2 + (y/e_1)**2 = 1
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
        #import time
        #t0 = time.time()
        
        #out = np.empty_like(modes)
        out = modes.copy()
        
        # diffuse terms: unflip the modes
        out[: modes.shape[0]//2] = \
                self.sym_ops.unflip_modes_Fourier(out[: out.shape[0]//2], apply_translation = False, inplace=True)
        
        # unit_cell terms: unflip the modes
        out[modes.shape[0]//2 :] = \
                self.sym_ops.unflip_modes_Fourier(out[out.shape[0]//2 :], apply_translation = True, inplace=True)
        
        # average 
        out_solid = np.mean(out, axis=0)
        
        #t1 = time.time()

        # propagate
        out_solid = np.fft.ifftn(out_solid)

        # reality
        out_solid.imag = 0

        #t2 = time.time()
        # finite support
        if self.voxel_number :
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
        
        #t3 = time.time()
        
        # store the latest guess for the object
        self.O = out_solid.copy()
        
        # propagate
        #t4 = time.time()
        out_solid = np.fft.fftn(out_solid)
        
        #t5 = time.time()
        # broadcast
        out[: out.shape[0]//2] = self.sym_ops.solid_syms_Fourier(out_solid, apply_translation=False, syms=out[: out.shape[0]//2])
        out[out.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(out_solid, apply_translation=True,  syms=out[out.shape[0]//2 :])

        self.iters += 1
        
        #t6 = time.time()
        #print(' sum | sqrt(I) - sqrt(Imap) | : ', self.Emod(out))
        #print('\nPsup')
        #print('total time        :', t6-t0)
        #print('unflip + mean time:', t1-t0)
        #print('ifftn time        :', t2-t1)
        #print('finite sup time   :', t3-t2)
        #print('imag copy time    :', t4-t3)
        #print('fftn time         :', t5-t4)
        #print('broadcast time    :', t6-t5)
        return out

    def Pmod(self, modes):
        #import time
        #t0 = time.time()
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
        #u = (np.fft.fftn(U, axes=(0,))/np.sqrt(U.shape[0]))[0].ravel()
        u = (np.sum(U, axis=0)/np.sqrt(U.shape[0])).ravel()
        
        y = np.abs(u) # np.abs(np.sum(U, axis=0) / np.sqrt(U.shape[0]))
        #print('\nchecking np.angle(U):', np.sum(np.angle(np.sum(U, axis=0)).ravel() \
                #                                           - np.angle(u)))
        
        #print('y sum:', np.sum(y))
        # look for forbidden reflections where e1_inf != 1
        forbidden   = (y < 10000*self.alpha) * (self.e1_inf == 0).ravel()
        
        #t1 = time.time()
        # project onto xp yp
        #-----------------------------------------------
        xp, yp = project_2D_Ellipse_arrays_cython(self.e0.ravel(), self.e1.ravel(), 
                                                  x, y, 
                                                  self.e0_inf.ravel(), self.e1_inf.ravel(), 
                                                  self.I0.ravel())
        
        #t2 = time.time()
        #print('yp sum:', np.sum(yp))
        # xp yp --> modes
        #-----------------------------------------------
        # d *= xp / x
        #############
        out = modes.copy()
        out[: modes.shape[0]//2] *= (xp / (x + self.alpha)).reshape(D[0].shape)
        
        #print('\nnp.sum(np.abs(xp[self.e1_inf==1])**2):', np.sum(np.abs(xp[self.e1_inf.ravel()==1])**2))
        #print('\nnp.sum(np.abs(out[: modes.shape[0]//2])**2):', np.sum(np.abs(out[: modes.shape[0]//2])**2))
        # u   *= yp / y
        # U[0] = u
        # U    = ifft(U)
        ################
        # for forbidden reflections be more careful 
        angle         = np.angle(u[forbidden])
        up            = np.empty_like(u)
        up[forbidden] = yp[forbidden] * np.exp(1J * angle)
        
        # for everything else just scale (this is much faster)
        up[~forbidden] = u[~forbidden] * yp[~forbidden] / (y[~forbidden] + self.alpha)
        
        #print('\nnp.sum((yp*np.sqrt(M * self.unit_cell_weighting) - self.amp)**2):',np.sum((yp * np.sqrt(4. * self.unit_cell_weighting).ravel()- self.amp.ravel())**2))
        #print('\nnp.sum((np.abs(up)*np.sqrt(self.unit_cell_weighting) - self.amp)**2):',np.sum((np.abs(up) * np.sqrt(4. * self.unit_cell_weighting).ravel()- self.amp.ravel())**2))
        #print('\nnp.sum((self.e1 - self.amp)**2):',np.sum((self.e1 - self.amp)**2))
        #print('\nnp.sum((np.abs(up)[self.e1_inf==1])**2):',np.sum((np.abs(up)[self.e1_inf.ravel()==1])**2))
        
        # un-rotate
        #out[modes.shape[0]//2 :] = np.sqrt(U.shape[0])*np.fft.ifftn(u, axes=(0,)).reshape(D.shape)
        out[modes.shape[0]//2 :] += (up - u).reshape(D[0].shape) / np.sqrt(U.shape[0])
        #print('\nnp.sum((np.abs(Bmodes)*np.sqrt(self.unit_cell_weighting) - self.amp)**2):',np.sum((np.abs(np.sum(out[modes.shape[0]//2 :], axis=0)) * np.sqrt(self.unit_cell_weighting)- self.amp)**2))
        
        # check
        #print('\nchecking np.abs(U):', np.sum(np.abs(np.sum(out[modes.shape[0]//2 :], axis=0)).ravel()  \
                #                                      / np.sqrt(U.shape[0]) - np.abs(up)))
        #print('\nchecking np.angle(U):', np.sum( (np.angle(up) - np.angle(u))[(np.abs(up)>1.0)*(np.abs(u)>1.0)] ))
        #print('\nnp.angle(DU)[0]     :', np.angle(up)[0] - np.angle(u)[0])
        #print('\nchecking np.angle(U):', np.sum((np.angle(np.sum(out[modes.shape[0]//2 :], axis=0)).ravel() \
                #                                           - np.angle(u))[(np.abs(up)>1.0)*(np.abs(u)>1.0)]))
        #print('\nchecking np.angle(U):', np.argmax((np.angle(np.sum(out[modes.shape[0]//2 :], axis=0)).ravel() \
                #                                           - np.angle(u))*(np.abs(up)>1.0)*(np.abs(u)>1.0)))
        # store the latest eMod, but just use one mode to speed this up
        #delta     = modes[4] - out[4]
        #self.eMod = self.l2norm(delta, out[4])
        
        #print('\n', self.eMod)
        #print('\neMod:', eMod)
        # check
        #print(' sum | sqrt(I) - sqrt(Imap) | : ', self.Emod(out))
        #print('Pmod -->  sum | modes - out | : ', np.sum( np.abs(modes - out) ))
        #t3 = time.time()
        #print('\nPmod:')
        #print('total time  :', t3-t0)
        #print('make xy time:', t1-t0)
        #print('ellipse time:', t2-t1)
        #print('apply mod time:', t3-t2)

        return out

    def Emod(self, modes):
        M         = self.Imap(modes)
        M         = self.mask * ( np.sqrt(M) - self.amp )**2
        eMod      = np.sum( M )
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
        info['eMod'] = [self.Emod(modes)]
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
