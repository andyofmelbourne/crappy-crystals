import numpy as np

from utils.disorder      import make_exp

def Mappings():
    """
    Defines the forward and inverse mappings
    for the solid unit to the complex wave at
    the detector.
    """
    def __init__(params):
        if params['crystal']['space_group'] == 'P1':
            import symmetry_operations.P1 as sym_ops 
        self.sym_ops = sym_ops

        # in general we have the inchorent mapping
        # and the inchoherent one (unit cell)
        # for now leave it
        self.N       = params['disorder']['n']
        self.exp     = make_exp(config['disorder']['sigma'], config['detector']['shape'])
        self.lattice = sym_ops.lattice(config['crystal']['unit_cell'], config['detector']['shape'])
        self.modes = lambda x : sym_ops.modes(x, config['crystal']['unit_cell'], config['detector']['shape'])

    def map(solid):
        modes = self.modes(solid)

        # unit cell mapping 
        diff  = self.N * self.exp * np.abs(self.lattice * np.sum(modes, axis=0))**2 
        
        # solid unit mapping
        diff += (1. - exp) * np.sum(np.abs(modes)**2, axis=0)
        return diff

    def imap(solid):


def phase(diff, params):
    """
    Just like normal iterative phase retrieval 
    but replace fft  --> mapping
    and         ifft --> inverse_mapping
    """

