import numpy as np
import afnumpy as ap
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
            import crappy_crystals.symmetry_operations.P1_gpu as sym_ops 
            print '\ncrystal space group: P1'
            self.sym_ops_obj = sym_ops.P1(params['crystal']['unit_cell'], params['detector']['shape'])
        elif params['crystal']['space_group'] == 'P212121':
            import crappy_crystals.symmetry_operations.P212121_gpu as sym_ops 
            self.sym_ops_obj = sym_ops.P212121(params['crystal']['unit_cell'], params['detector']['shape'])
            print '\ncrystal space group: P212121'
        
        self.sym_ops = self.sym_ops_obj.solid_syms_Fourier

        # in general we have the inchorent mapping
        # and the inchoherent one (unit cell)
        # for now leave it
        self.N          = params['disorder']['n']
        self.exp        = make_exp(params['disorder']['sigma'], params['detector']['shape'])
        self.lattice    = sym_ops.lattice(params['crystal']['unit_cell'], params['detector']['shape'])
        self.DB         = None

        self.exp     = ap.array(self.exp)
        self.lattice = ap.array(self.lattice)
    
    def modes(self, solid_syms):
        if self.DB is None :
            self.DB = ap.zeros((2,) + solid_syms.shape[1 :], dtype=solid_syms.real.dtype)
        
        # diffuse term (incoherent sum)
        self.DB[0] = (1. - self.exp) * ap.sum(ap.abs(solid_syms)**2, axis=0)
        # brag term (coherent sum)
        B     = ap.sum(solid_syms, axis=0)
        self.DB[1] = self.N * self.exp * self.lattice * (B.conj() * B).real
        return self.DB

    def make_diff(self, solid = None, solid_syms = None):
        if solid_syms is None :
            solid_syms = self.sym_ops(solid)
        
        modes = self.modes(solid_syms)
        
        diff = ap.sum(modes, axis=0)
        return diff


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

