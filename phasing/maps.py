import numpy as np
import sys

import crappy_crystals
import crappy_crystals.utils.disorder
import crappy_crystals.utils.l2norm
from   crappy_crystals.utils.disorder import make_exp
from   crappy_crystals.utils.l2norm   import l2norm


class Mappings():
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
        elif params['crystal']['space_group'] == 'P212121':
            import crappy_crystals.symmetry_operations.P212121 as sym_ops 
            print '\ncrystal space group: P212121'
        self.sym_ops = sym_ops

        # in general we have the inchorent mapping
        # and the inchoherent one (unit cell)
        # for now leave it
        self.N          = params['disorder']['n']
        self.exp        = make_exp(params['disorder']['sigma'], params['detector']['shape'])
        self.lattice    = sym_ops.lattice(params['crystal']['unit_cell'], params['detector']['shape'])
        self.solid_syms = lambda x : sym_ops.solid_syms(x, params['crystal']['unit_cell'], \
                                                        params['detector']['shape'])
    
    def modes(self, solid_syms):
        modes = np.zeros((solid_syms.shape[0] + 1,) + solid_syms.shape[1 :], dtype=solid_syms.dtype)
        
        # solid unit mapping
        modes[:-1] = np.sqrt(1. - self.exp) * solid_syms
        
        # unit cell mapping 
        modes[-1]  = np.sqrt(self.N * self.exp) * self.lattice * np.sum(solid_syms, axis=0)
        return modes

    def make_diff(self, solid = None, solid_syms = None):
        if solid_syms is None :
            solid_syms = self.solid_syms(solid)

        modes = self.modes(solid_syms)

        diff = np.sum(np.abs(modes)**2, axis=0)
        return diff

def HIO_(exit, Pmod, Psup, beta=1.):
    out = Pmod(exit)
    out = exit + beta * Psup( (1+1/beta)*out - 1/beta * exit ) - beta * out  
    return out

def DM_(psi, Pmod, Psup, beta=0.7):
    """
    psi_j+1 = psi_j - Ps psi_j - Pm psi_j
            + b(1+1/b) Ps Pm psi_j
            - b(1-1/b) Pm Ps psi_j
    """
    psi_M = psi.copy()
    psi_M = Pmod(psi_M)
    psi_S = Psup(psi)
    psi  -= psi_M + psi_S
    psi  += Psup(beta * (1. + 1. / beta) * psi_M)
    psi_S = Pmod(psi_S)
    psi  -= beta * (1. - 1. / beta) * psi_S
    return psi

def DM_to_sol_(psi, Pmod, Psup, beta):
    psi_M = psi.copy()
    psi_M = Pmod(psi_M)
    psi_M = (1. + 1./beta) * psi_M - 1./beta * psi
    psi_M = Psup(psi_M)
    return psi_M

def Pmod_(modes, diff, M, good_pix, alpha = 1.0e-10):
    
    modes = modes * (~good_pix + good_pix * np.sqrt(diff) / (np.sqrt(M) + alpha))
    
    return modes


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

