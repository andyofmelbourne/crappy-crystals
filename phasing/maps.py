import numpy as np
import sys

import crappy_crystals
import crappy_crystals.utils.disorder
from   crappy_crystals.utils.disorder import make_exp
#import crappy_crystals.utils.l2norm
#from   crappy_crystals.utils.l2norm   import l2norm


import phasing_3d
from phasing_3d.src.mappers import Modes
from phasing_3d.src.mappers import choose_N_highest_pixels, isValid

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
            modes = np.fft.fftn(args['O'])
        else :
            print 'initialising object with random numbers'
            modes = np.random.random(I.shape).astype(args['c_dtype'])
        
        # initialise the mask, alpha value and amp
        #-----------------------------------------------
        self.mask = 1
        if isValid('mask', args):
            self.mask = args['mask']
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']
        
        self.I_norm = (self.mask * I).sum()
        self.amp    = np.sqrt(I.astype(args['dtype']))
        
        # define the support projection
        #-----------------------------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
        else :
            self.voxel_number = False
            self.S    = args['support']
        
        self.support = None
        if isValid('support', args):
            self.support = args['support']

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




class Mappings_old():
    """
    There are two layers of mappings we have to deal with here:
    
    First there is the 'crystal' mapping which takes the solid 
    unit and each of its symmetry related partners to the detector.
    These I will call solid_syms:
    Mappings.solid_syms o --> O(R1 . q), O(R2 . q) ...
    
    Then there is the mapping from the solid_syms to the coherent
    modes:
    Mappings.modes: O(R1 . q), O(R2 . q), ... --> psi1, psi2 ...
    such that:      I = |psi1|**2 + |psi2|**2 + ...
    """
    def __init__(self, params):
        if params['crystal']['space_group'] == 'P1':
            import crappy_crystals.symmetry_operations.P1 as sym_ops 
            print '\ncrystal space group: P1'
            self.sym_ops_obj = sym_ops.P1(params['crystal']['unit_cell'], params['detector']['shape'])
        elif params['crystal']['space_group'] == 'P212121':
            import crappy_crystals.symmetry_operations.P212121 as sym_ops 
            self.sym_ops_obj = sym_ops.P212121(params['crystal']['unit_cell'], params['detector']['shape'])
            print '\ncrystal space group: P212121'
        
        self.space_group = params['crystal']['space_group']
        
        self.sym_ops = self.sym_ops_obj.solid_syms_Fourier

        # in general we have the inchorent mapping
        # and the inchoherent one (unit cell)
        # for now leave it
        self.N          = params['disorder']['n']
        self.exp        = make_exp(params['disorder']['sigma'], params['detector']['shape'])
        self.lattice    = sym_ops.lattice(params['crystal']['unit_cell'], params['detector']['shape'])
        self.solid_syms = lambda x : sym_ops.solid_syms(x)
        self.DB         = None
    
    def modes(self, solid_syms, return_DandB = False):
        if self.DB is None :
            self.DB = np.zeros((2,) + solid_syms.shape[1 :], dtype=solid_syms.real.dtype)
        
        # diffuse term (incoherent sum)
        D     = np.sum((solid_syms.conj() * solid_syms).real, axis=0)
        self.DB[0] = (1. - self.exp) * D
        
        # brag term (coherent sum)
        B     = np.sum(solid_syms, axis=0)
        self.DB[1] = self.N * self.exp * self.lattice * (B.conj() * B).real

        if return_DandB :
            return self.DB, solid_syms, B
        else :
            return self.DB

    def make_diff(self, solid = None, solid_syms = None, return_DandB = False):
        if solid_syms is None :
            solid_syms = self.sym_ops(solid)
        
        if return_DandB :
            modes, D, B = self.modes(solid_syms, return_DandB = return_DandB)
        else :
            modes       = self.modes(solid_syms, return_DandB = return_DandB)
        
        diff = np.sum(modes, axis=0)
        
        if return_DandB :
            return diff, D, B
        else :
            return diff
    
    def merge_solids_unit_cell(self, solids, unit_cell, iters = 10):
        if self.space_group == 'P1' :
            return (solids+unit_cell) / 2.
         
        b = np.array([4*solids[0].real, unit_cell.real])
        
        Adot  = lambda x : np.array([4*x, np.sum(self.sym_ops_obj.solid_syms_real(x), axis=0)])
        ATdot = lambda x : 4*x[0] + np.sum(self.sym_ops_obj.solid_syms_real(x[1]), axis=0)
        
        cgls = Cgls(Adot, b, ATdot)
        solid_unit_retrieved = cgls.cgls(iters)
        print '\nCGLS residual:', cgls.e_res[0], '-->', cgls.e_res[-1]
        #print 'error:', np.sum((solid_unit - solid_unit_retrieved)**2)
        return solid_unit_retrieved.astype(solids.dtype)

class Cgls(object):
    """Run the cgls algorithm in general given the functions Adot and ATdot and the bvector.
    
    Solves A . x = b 
    given routines for A . x' and AT . b'
    and the bvector
    where x and b may be any numpy arrays."""

    def __init__(self, Adot, bvect, ATdot = None, imax = 10**5, e_tol = 1.0e-10, x0 = None):
        self.Adot   = Adot
        self.ATdot  = ATdot
        self.bvect  = bvect
        self.iters  = 0
        self.imax   = imax
        self.e_tol  = e_tol
        self.e_res  = []
        if x0 is None :
            if self.ATdot is None :
                self.x = Adot(bvect)
            else :
                self.x = ATdot(bvect)
            self.x.fill(0.0)
        else :
            self.x = x0

    def cgls(self, iterations = None):
        """Iteratively solve the linear equations using the steepest descent algorithm.
        
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again.
        
        AT . A . x = AT . b   solves || A . x - b ||_min(x)
        
        d_0 = r_0 = AT . b - AT . A . x_0
        
        for i: 0 --> iters or while ||r_i|| / ||r_0|| < e_tol :
            alpha_i  = ||r_i|| / || A . d ||
            x_i+1    = x_i + alpha_i d_i
            r_i+1    = r_i - alpha_i AT . A . d_i
            beta     = r_i - ||r_i+1|| / ||r_i||
            d_i+1    = r_i+1 + beta d_i 
        """
        if self.iters == 0 :
            self.r         = self.ATdot(self.bvect) - self.ATdot(self.Adot(self.x))
            self.d         = self.r.copy()
            self.rTr_new   = np.sum(self.r**2)
            self.rTr_0     = self.rTr_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        for i in range(iterations):
            Ad     = self.Adot(self.d)
            alpha  = self.rTr_new / np.sum(Ad * Ad)
            self.x = self.x + alpha * self.d
            #
            if self.iters % 1000 == 0 :
                self.r = self.ATdot(self.bvect) - self.ATdot(self.Adot(self.x))
            else :
                self.r = self.r - alpha * self.ATdot(Ad)
            #
            rTr_old        = self.rTr_new.copy()
            self.rTr_new   = np.sum(self.r**2)
            beta           = self.rTr_new / rTr_old
            self.d         = self.r + beta * self.d
            #
            self.iters = self.iters + 1
            self.e_res.append(np.sqrt(self.rTr_new))
            if self.iters > self.imax : 
                #print 'cgls: reached maximum iterations', self.imax
                return self.x
            #if self.rTr_new < self.e_tol**2 * self.rTr_0:
            #    #print 'cgls: error tolerance achieved at', i+1, 'iterations'
            #    return self.x
        #
        return self.x

def update_progress(progress, algorithm, i, emod, esup):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5} {6} {7}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, emod, esup, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
    sys.stdout.flush()

