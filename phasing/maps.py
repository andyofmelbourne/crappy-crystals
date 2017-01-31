from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys

import crappy_crystals
import crappy_crystals.utils.disorder
from   crappy_crystals.utils.disorder import make_exp
#import crappy_crystals.phasing.symmetry_operations as symmetry_operations 
from . import symmetry_operations 
#import crappy_crystals.utils.l2norm
#from   crappy_crystals.utils.l2norm   import l2norm


import phasing_3d
from phasing_3d.src.mappers import Modes
from phasing_3d.src.mappers import isValid


import pyximport; pyximport.install()
from .ellipse_2D_cython import project_2D_Ellipse_cython

def get_sym_ops(params):

    if params['crystal']['space_group'] == 'P1':
        print('\ncrystal space group: P1')
        sym_ops = \
            symmetry_operations.P1(params['crystal']['unit_cell'], params['detector']['shape'])

    elif params['crystal']['space_group'] == 'P212121':
        print('\ncrystal space group: P212121')
        sym_ops = \
            symmetry_operations.P212121(params['crystal']['unit_cell'], params['detector']['shape'])

    return sym_ops


class Mapper_naive():

    def __init__(self, I, **args):
        """
        For mapper naive we just have a single object.

        The support projection is:
            Psup O = F . S . F-1 . O

        And the modulus projection is:
            Pmod O = O sqrt{ I / M }

        Where M is the forward mapping (self.Imap) of O
        to the diffraction intensity:
            M = (1 - gaus) sum_i | O_i |^2 +
                N * L * gaus * | sum_i O_i |^2 


        Parameters
        ----------
        I : numpy.ndarray, (N, M, K)
            Merged diffraction patterns to be phased. 
        
            N : the number of pixels along slowest scan axis of the detector
            M : the number of pixels along slow scan axis of the detector
            K : the number of pixels along fast scan axis of the detector
        
        O : numpy.ndarray, (N, M, K) 
            The real-space scattering density of the object such that:
                I = |F[O]|^2
            where F[.] is the 3D Fourier transform of '.'.     
        
        dtype : np.float32 or np.float64
            Determines the numerical precision of the calculation. 

        c_dtype : np.complex64 or np.complex128
            Determines the numerical precision of the complex variables. 

        support : (numpy.ndarray, None or int), optional (N, M, K)
            Real-space region where the object function is known to be zero. 
            If support is an integer then the N most intense pixels will be kept at
            each iteration.

        voxel_number : None or int, optional
            If int, then the voxel number support is used. If support is not None or False
            then the voxel number support is used within the sample support bounds.
        
        mask : numpy.ndarray, (N, M, K), optional, default (1)
            The valid detector pixels. Mask[i, j, k] = 1 (or True) when the detector pixel 
            i, j, k is valid, Mask[i, j, k] = 0 (or False) otherwise.
        
        alpha : float, optional, default (1.0e-10)
            A floating point number to regularise array division (prevents 1/0 errors).
        
        disorder['sigma'] : float
            The standard deviation of the isotropic displacement of the solid unit within
            the crystal.

        detector['shape'] : tupple
            The shape of the detector (3D).

        crystal['unit_cell'] : tupple
            The dimensions of the unit cell

        Returns
        -------
        O : numpy.ndarray, (U, V, K) 
            The real-space object function after 'iters' iterations of the ERA algorithm.
        
        info : dict
            contains diagnostics:
                
                'I'     : the diffraction pattern corresponding to object above
                'eMod'  : the modulus error for each iteration:
                          eMod_i = sqrt( sum(| O_i - Pmod(O_i) |^2) / I )
                'eCon'  : the convergence error for each iteration:
                          eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        """
        # dtype
        #-----------------------------------------------
        if isValid('dtype', args) :
            dtype = args['dtype']
        else :
            dtype = np.float64
        
        if isValid('c_dtype', args) :
            c_dtype = args['c_dtype']
        else :
            c_dtype = np.complex128
        
        # initialise the object
        #-----------------------------------------------
        if isValid('O', args):
            modes = np.fft.fftn(args['O'])
        else :
            print('initialising object with random numbers')
            modes = np.random.random(I.shape).astype(c_dtype)
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = 1
        if isValid('mask', args):
            self.mask = args['mask']
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']
        
        self.I_norm = (self.mask * I).sum()
        self.amp    = np.sqrt(I.astype(dtype))
        
        # define the support projection
        #-----------------------------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
            self.support = None
        else :
            self.voxel_number = False
            #
            self.support = args['support']
            self.S       = self.support.copy()
        
        # make the unit cell and diffuse weightings
        #-----------------------------------------------
        self.sym_ops = get_sym_ops(args)
        
        N          = args['disorder']['n']
        exp        = make_exp(args['disorder']['sigma'], args['detector']['shape'])
        lattice    = symmetry_operations.lattice(args['crystal']['unit_cell'], args['detector']['shape'])
        #self.solid_syms = lambda x : sym_ops.solid_syms(x)
        
        self.unit_cell_weighting = N * lattice * exp
        self.diffuse_weighting   = (1. - exp)
        
        self.modes = modes
         
    def object(self, modes):
        out = np.fft.ifftn(modes)
        return out

    def Imap(self, modes):
        # 'modes' is just our object in Fourier space
        Us = self.sym_ops.solid_syms_Fourier(modes)
        U  = np.sum(Us, axis=0)
        
        I  = self.diffuse_weighting   * np.sum( (Us * Us.conj()).real, axis=0)
        I += self.unit_cell_weighting * (U * U.conj()).real
        return I
    
    def Psup(self, modes):
        out = modes.copy()
        out = np.fft.ifftn(out)

        if self.voxel_number :
            self.S = choose_N_highest_pixels( (out * out.conj()).real, self.voxel_number, support = self.support)

        out *= self.S
        out = np.fft.fftn(out)
        return out

    def Pmod(self, modes):
        out = modes.copy()
        M   = self.Imap(out)
        out = pmod_naive(self.amp, M, modes, self.mask, alpha = self.alpha)
        return out
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = np.sum( self.mask * ( np.sqrt(M) - self.amp )**2 )
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        out['support'] = self.S
        out['I']       = self.Imap(modes)
        return out

    def l2norm(self, delta, array0):
        num = 0
        den = 0
        num += np.sum( (delta * delta.conj()).real ) 
        den += np.sum( (array0 * array0.conj()).real ) 
        return np.sqrt(num / den)
     

def pmod_naive(amp, M, O, mask = 1, alpha = 1.0e-10):
    out  = mask * O * amp / np.sqrt(M + alpha)
    out += (1 - mask) * O
    return out


class Mapper_ellipse():

    def __init__(self, I, **args):
        """
        """
        print('Hello I am mapper ellipse')
        # dtype
        #-----------------------------------------------
        if isValid('dtype', args) :
            dtype = args['dtype']
        else :
            dtype = np.float64
        
        if isValid('c_dtype', args) :
            c_dtype = args['c_dtype']
        else :
            c_dtype = np.complex128

        # initialise the object
        #-----------------------------------------------
        if isValid('O', args):
            O = np.fft.fftn(args['O'])
        else :
            print('initialising object with random numbers')
            O = np.random.random(I.shape).astype(c_dtype)

        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = 1
        if isValid('mask', args):
            self.mask = args['mask']
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']
        
        self.I_norm = (self.mask * I).sum()
        self.amp    = np.sqrt(I.astype(dtype))
        
        # define the support projection
        #-----------------------------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
            self.support = None
            self.S       = None
        else :
            self.voxel_number = False
            #
            self.support = args['support']
            self.S       = self.support.copy()
        
        # make the unit cell and diffuse weightings
        #-----------------------------------------------
        self.sym_ops = get_sym_ops(args)
        
        N          = args['disorder']['n']
        exp        = make_exp(args['disorder']['sigma'], args['detector']['shape'])
        lattice    = symmetry_operations.lattice(args['crystal']['unit_cell'], args['detector']['shape'])
        self.unit_cell = args['crystal']['unit_cell']

        self.unit_cell_weighting = N * lattice * exp
        self.diffuse_weighting   = (1. - exp)
        
        self.modes = np.zeros( (2 * self.sym_ops.syms.shape[0],) + self.sym_ops.syms.shape[1:], O.dtype)
        # diffuse terms
        self.modes[:self.modes.shape[0]//2] = self.sym_ops.solid_syms_Fourier(O, apply_translation = False)
        # unit cell terms
        self.modes[self.modes.shape[0]//2:] = self.sym_ops.solid_syms_Fourier(O, apply_translation = True)

        print('eMod(modes0):', self.Emod(self.modes))

        # Ellipse axes
        #-----------------------------------------------
        # Here we have :
        # (x / e0)**2 + (y / e1)**2 = 1 , where
        # 
        # e0 = sqrt{ self.diffuse_weighting / I } and
        # 
        # e1 = sqrt{ self.unit_cell_weighting / I }
        #-----------------------------------------------
        
        # floating point tolerance for 1/x (log10)  
        tol = 100. #1.0e+100
        
        # check for numbers close to infinity in sqrt(I / self.diffuse_weighting)
        m     = self.diffuse_weighting <= 0.0
        m[~m] = 0.5 * (np.log10(I[~m]) - np.log10(self.diffuse_weighting[~m])) > tol
        
        self.e0_inf   = m.copy()
        self.e0       = np.zeros_like(self.diffuse_weighting)
        self.e0[~m]   = np.sqrt(I[~m]) / np.sqrt(self.diffuse_weighting[~m])
               
        # check for numbers close to infinity in sqrt(I / self.unit_cell_weighting)
        m     = self.unit_cell_weighting <= 0.0 
        m[~m] = 0.5 * (np.log10(I[~m]) - np.log10(self.unit_cell_weighting[~m])) > tol
              
        self.e1_inf = m.copy()
        self.e1     = np.zeros_like(self.unit_cell_weighting)
        self.e1[~m] = np.sqrt(I[~m]) / np.sqrt(self.sym_ops.syms.shape[0] * self.unit_cell_weighting[~m])

        self.iters = 0
         
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
            #self.S = choose_N_highest_pixels( (out * out.conj()).real, self.voxel_number, \
                    #        support = self.support, mapper = self.sym_ops.solid_syms_real)
            # try using the crystal mapping instead of the unit-cell mapping
            self.S = choose_N_highest_pixels( (out * out.conj()).real, self.voxel_number, \
                    support = self.support, mapper = self.sym_ops.solid_syms_cryst_real)

        out *= self.S

        # reality
        out.imag = 0

        # propagate
        out = np.fft.fftn(out)

        # broadcast
        modes_out = np.empty_like(self.modes)
        modes_out[: modes_out.shape[0]//2] = self.sym_ops.solid_syms_Fourier(out, apply_translation=False)
        modes_out[modes_out.shape[0]//2 :] = self.sym_ops.solid_syms_Fourier(out, apply_translation=True)

        self.iters += 1
        
        return modes_out

    def Pmod(self, modes):
        #return modes.copy()
        #out = modes.copy()
        #M   = self.Imap(out)
        #out = pmod_naive(self.amp, M, modes, self.mask, alpha = self.alpha)
        ###################################

        U  = modes[modes.shape[0]//2 :]
        D  = modes[: modes.shape[0]//2]

        # make x
        #-----------------------------------------------
        x = np.sqrt(np.sum( (D * D.conj()).real, axis=0))
        
        # make y
        #-----------------------------------------------
        # rotate the unit cell modes
        # eg. for N = 2 : Ut = np.array([[1., 1.], [-1., 1.]], dtype=np.float64) / np.sqrt(2.)

        Ut = make_unitary_transform(modes.shape[0]//2)

        u = np.dot(Ut, U.reshape((U.shape[0], -1)))
        
        y = np.abs(np.sum(U, axis=0) / np.sqrt(U.shape[0]))
        
        tol = 1.0e-10
        # project onto xp yp
        #-----------------------------------------------
        #assert np.all(self.e0[~self.e0_inf] > 0)
        #assert np.all(self.e1[~self.e1_inf] > 0)
        xp = np.empty_like(x)
        yp = np.empty_like(x)
        it = np.nditer([self.e0, self.e1, x, y, self.e0_inf, self.e1_inf, xp, yp],
                        op_flags = [['readonly'], ['readonly'], ['readonly'], ['readonly'],
                                    ['readonly'],['readonly'],['writeonly', 'no_broadcast'],
                                    ['writeonly', 'no_broadcast']])
        for e0, e1, xi, yi, e0_inf, e1_inf, uu, vv in it:
            uu[...], vv[...] = project_2D_Ellipse_cython(e0, e1, xi, yi, e0_inf, e1_inf)

            # check
            #if (not e0_inf) and (not e1_inf) :
            #    err = abs((uu[...] / e0)**2 + (vv[...] / e1)**2 - 1.)
            #    if err > tol :
            #        print('e0, e1, xi, yi, xp, yp, err:', e0, e1, xi, yi, uu[...], vv[...], err)

        
        # xp yp --> modes
        #-----------------------------------------------
        rx = xp / (x + self.alpha)
        out = modes.copy()
        out[: modes.shape[0]//2] *= rx
        
        ry = yp / (y + self.alpha)
        u[0] *= ry.ravel()
        
        # un rotate the y's
        out[modes.shape[0]//2 :] = np.dot(Ut.T, u).reshape(U.shape)

        # check
        #print(' sum | sqrt(I) - sqrt(Imap) | : ', self.Emod(out))
        
        return out
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = np.sum( self.mask * ( np.sqrt(M) - self.amp )**2 )
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        out['support'] = self.S
        out['I']       = self.Imap(modes)
        return out

    def l2norm(self, delta, array0):
        num = 0
        den = 0
        for i in range(delta.shape[0]):
            num += np.sum( (delta[0] * delta[0].conj()).real ) 
            den += np.sum( (array0[0] * array0[0].conj()).real ) 
        return np.sqrt(num / den)

    def scans_cheshire(self, solid):
        """
        scan the solid unit through the cheshire cell 
        until the best agreement with the data is found.
        """
        
        s = phasing_3d.utils.merge.centre(solid)
        I, J, K = self.unit_cell
        I //= 2
        J //= 2
        K //= 2
        modes = np.empty_like(self.modes)
        step = 4
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
    
    if support is not None :
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
            print('e==0, exiting...')
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
    
    print('number of pixels in support:', np.sum(S), i, s, e)
    return S
