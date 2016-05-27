import numpy as np
import sys
import os
import ConfigParser

sys.path.append(os.path.abspath('.'))

from utils.disorder      import make_exp
from utils.io_utils      import parse_parameters
from utils.io_utils      import parse_cmdline_args_phasing
from utils.io_utils      import read_input_h5
from utils.io_utils      import write_output_h5

from gpu.phasing.maps import *

class Mappings_gpu(Mappings):
    def __init__(self, duck, support, amp, good_pix, params):
        Mappings.__init__(self, duck, support, amp, good_pix, params)
        
        if params['crystal']['space_group'] == 'P1':
            import symmetry_operations.P1 as sym_ops 
        #elif params['crystal']['space_group'] == 'P212121':
        #    import symmetry_operations.P212121 as sym_ops 
        self.sym_ops = sym_ops
        
        exp          = make_exp(params['disorder']['sigma'], params['detector']['shape'])
        self.exp     = pyopencl.array.to_device(self.queue, np.ascontiguousarray(exp.astype(self.amp.dtype)))
        self.N       = params['disorder']['n']
        lattice      = sym_ops.lattice(params['crystal']['unit_cell'], params['detector']['shape'])
        self.lattice = pyopencl.array.to_device(self.queue, np.ascontiguousarray(lattice.astype(np.int8)))
    
    def make_diff(self, psi_F):
        self.dummy_real2 = abs(psi_F)**2
        diff  = (1. - self.exp) * self.dummy_real2
        diff += self.N * self.exp * self.lattice * self.dummy_real2
        return diff
    
    def Pmod(self, psi):
        self.plan.execute(psi.data)
        
        self.dummy_real2 = self.make_diff(psi)
        
        psi = self._Pmod(psi, self.dummy_real2)
        
        self.plan.execute(psi.data, inverse=True)
        return psi

    def Psup(self, psi):
        # apply support
        psi *= self.support
        
        # apply positivity & reality
        self.dummy_real.fill(0.0)
        
        pyopencl.array.maximum(psi.real, self.dummy_real, out = self.dummy_real2)
        psi = self.dummy_real2.astype(psi.dtype)
        return psi

    def ERA(self, psi):
        psi = self.Psup(psi)
        psi = self.Pmod(psi)
        return psi


def l2norm_gpu(array1, array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot = pyopencl.array.sum(abs(array1)**2).get()
    tot = np.sqrt(pyopencl.array.sum(abs(array1-array2)**2).get()/tot)
    return tot


def phase(I, solid_support, params, good_pix = None, solid_known = None):
    """
    """
    if good_pix is None :
        good_pix = I > -1
    

    duck = solid_support * (np.random.random(solid_support.shape) + 0.0J)
    
    maps = Mappings_gpu(duck, support, np.sqrt(I), good_pix, params)
    
    maps.duck_known = pyopencl.array.to_device(maps.queue, np.ascontiguousarray(solid_known.astype(maps.duck.dtype)))

    maps.dummy_comp = maps.duck_known.copy()
    maps.plan.execute(maps.dummy_comp.data)
    M = maps.make_diff(maps.dummy_comp)
    
    print 'l2norm_gpu(np.sqrt(I), pyopencl.clmath.sqrt(M))',l2norm_gpu(maps.amp, pyopencl.clmath.sqrt(M))
    
    e_mod = []
    e_sup = []
    e_fid = []
    
    iters = params['phasing']['ERA']
    for i in range(iters):
        maps.duck = maps.ERA(maps.duck)
        
        # calculate the fidelity and modulus error
        maps.dummy_comp = maps.duck.copy()
        maps.dummy_comp = maps.Psup(maps.dummy_comp)
        maps.plan.execute(maps.dummy_comp.data)
        M = maps.make_diff(maps.dummy_comp)
        e_mod.append(l2norm_gpu(maps.amp, pyopencl.clmath.sqrt(M)))
        
        if solid_known is not None :
            e_fid.append(l2norm_gpu(maps.duck_known, maps.duck))
        else :
            e_fid.append(-1)
        
        update_progress(i / max(1.0, float(iters-1)), 'ERA', i, e_mod[-1], e_fid[-1])
    print '\n'
    return maps.duck.get(), M.get(), e_mod, e_fid


if __name__ == "__main__":
    args = parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, solid_known, params = read_input_h5(args.input)
    
    solid_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, solid_known = solid_known)
    
    # write the h5 file 
    write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, solid_known, solid_ret, emod, efid)
