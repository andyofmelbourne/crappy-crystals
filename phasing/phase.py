"""
o                                            the solid unit
os = o(R1 . x + T1), o(R2 . x + T2), ...     the symmetry related coppies of o
Os = e^{T1} O(R1 . q), e^{T2} O(R2 . q), ... the symmetry related coppies of o at the detector

modes = 

u = sum_i o(Ri . x + Ti)    the unit cell
"""
import numpy as np

from utils.disorder      import make_exp
import bagOfns as bg

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
            import symmetry_operations.P1 as sym_ops 
        self.sym_ops = sym_ops

        # in general we have the inchorent mapping
        # and the inchoherent one (unit cell)
        # for now leave it
        self.N          = params['disorder']['n']
        self.exp        = make_exp(params['disorder']['sigma'], params['detector']['shape'])
        self.lattice    = sym_ops.lattice(params['crystal']['unit_cell'], params['detector']['shape'])
        self.solid_syms = lambda x : sym_ops.solid_syms(x, params['crystal']['unit_cell'], params['detector']['shape'])

    def isolid_syms(self, solid_syms):
        """
        if the space group is P1 then
        this is really easy
        """
        if self.sym_ops.name == 'P1':
            solid = np.fft.ifftn(solid_syms[0])
        return solid

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


def _Pmod(modes, diff, M, alpha = 1.0e-10):
    
    modes = modes * np.sqrt(diff) / (np.sqrt(M) + alpha)
    
    return modes


def phase(I, solid_support, params, solid_known = None):
    """
    """
    maps = Mappings(params)
    
    def Pmod(x):
        """
        O --> solid_syms
        _Pmod
        modes --> O
        """
        solid_syms = maps.solid_syms(x)
        solid_syms = _Pmod(solid_syms, I, maps.make_diff(solid_syms = solid_syms))
        x          = maps.isolid_syms(solid_syms)
        return x

    def Psup(x):
        # apply support
        y = x * solid_support
        
        # apply reality
        y.imag = 0.0

        # apply positivity
        y[np.where(y<0)] = 0.0
        return y

    #Psup = lambda x : (x * solid_support).real + 0.0J
    
    ERA = lambda x : Psup(Pmod(x))
    HIO = lambda x : bg.HIO(x.copy(), Pmod, Psup, beta=1.)

    iters = 500
    e_mod = []
    e_sup = []
    e_fid = []

    print 'alg: progress iteration modulus error fidelty'
    x = np.random.random(solid_support.shape) + 0.0J
    for i in range(iters):
        x = HIO(x)
        
        # calculate the fidelity and modulus error
        M = maps.make_diff(solid = x)
        e_mod.append(bg.l2norm(np.sqrt(I), np.sqrt(M)))
        e_sup.append(bg.l2norm(x, Psup(x)))
        if solid_known is not None :
            e_fid.append(bg.l2norm(solid_known + 0.0J, x))
        else :
            e_fid.append(-1)
        
        bg.update_progress(i / max(1.0, float(iters-1)), 'HIO', i, e_mod[-1], e_fid[-1])
    
    iters = 100
    for i in range(iters):
        x = ERA(x)
        
        # calculate the fidelity and modulus error
        M = maps.make_diff(solid = x)
        e_mod.append(bg.l2norm(np.sqrt(I), np.sqrt(M)))
        e_sup.append(bg.l2norm(x, Psup(x)))
        if solid_known is not None :
            e_fid.append(bg.l2norm(solid_known + 0.0J, x))
        else :
            e_fid.append(-1)
        
        bg.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, e_mod[-1], e_fid[-1])
    print '\n'

    return x, M
