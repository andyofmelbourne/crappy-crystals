import afnumpy 
import afnumpy.fft
import numpy as np
import sys

import crappy_crystals
import crappy_crystals.utils.disorder
from   crappy_crystals.utils.disorder import make_exp
#import crappy_crystals.utils.l2norm
#from   crappy_crystals.utils.l2norm   import l2norm


import phasing_3d
from phasing_3d.src.mappers_gpu import Modes
from phasing_3d.src.mappers_gpu import choose_N_highest_pixels, isValid

import symmetry_operations

def get_sym_ops(params):

    if params['crystal']['space_group'] == 'P1':
        print '\ncrystal space group: P1'
        sym_ops = \
            symmetry_operations.P1(params['crystal']['unit_cell'], params['detector']['shape'])

    elif params['crystal']['space_group'] == 'P212121':
        print '\ncrystal space group: P212121'
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

        # initialise the object
        #-----------------------------------------------
        if isValid('O', args):
            modes = afnumpy.fft.fftn(afnumpy.array(args['O']))
        else :
            print 'initialising object with random numbers'
            modes = afnumpy.random.random(I.shape).astype(args['c_dtype'])
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = 1
        if isValid('mask', args):
            if args['mask'] is not 1 :
                self.mask = afnumpy.array(args['mask'].astype(np.bool))
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']
        
        if isValid('mask', args) :
            self.I_norm = (args['mask'] * I).sum()
        else :
            self.I_norm = I.sum()
        
        self.amp   = afnumpy.sqrt(afnumpy.array(I.astype(args['dtype'])))
        
    
        # define the support projection
        # -----------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
        else :
            self.voxel_number = False
            self.S    = afnumpy.array(args['support'])
        
        self.support = None
        if isValid('support', args):
            self.support = afnumpy.array(args['support'])

        self.modes = modes

        # make the unit cell and diffuse weightings
        #-----------------------------------------------
        self.sym_ops = get_sym_ops(args)
        
        N          = args['disorder']['n']
        exp        = make_exp(args['disorder']['sigma'], args['detector']['shape'])
        lattice    = symmetry_operations.lattice(args['crystal']['unit_cell'], args['detector']['shape'])
        #self.solid_syms = lambda x : sym_ops.solid_syms(x)

        self.unit_cell_weighting = afnumpy.array(N * lattice * exp)
        self.diffuse_weighting   = afnumpy.array(1. - exp)
        
        self.modes = modes
         
    def object(self, modes):
        out = np.array(afnumpy.fft.ifftn(modes))
        return out

    def Imap(self, modes):
        # 'modes' is just our object in Fourier space
        Us = self.sym_ops.solid_syms_Fourier(modes)
        U  = afnumpy.sum(Us, axis=0)
        
        I  = self.diffuse_weighting   * np.sum( (Us * Us.conj()).real, axis=0)
        I += self.unit_cell_weighting * (U * U.conj()).real
        return I
    
    def Psup(self, modes):
        out = modes.copy()
        out = afnumpy.fft.ifftn(out)
        
        if self.voxel_number :
            self.S = choose_N_highest_pixels( (out * out.conj()).real, self.voxel_number, support = self.support)
        
        out *= self.S
        out = afnumpy.fft.fftn(out)
        return out

    def Pmod(self, modes):
        out = modes.copy()
        M   = self.Imap(out)
        out = pmod_naive(self.amp, M, modes, self.mask, alpha = self.alpha)
        return out
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = afnumpy.sum( self.mask * ( afnumpy.sqrt(M) - self.amp )**2 )
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
        num += afnumpy.sum( (delta * delta.conj()).real ) 
        den += afnumpy.sum( (array0 * array0.conj()).real ) 
        return afnumpy.sqrt(num / den)
     

def pmod_naive(amp, M, O, mask = 1, alpha = 1.0e-10):
    out  = mask * O * amp / afnumpy.sqrt(M + alpha)
    out += (1 - mask) * O
    return out




