import numpy as np

name = 'P1'

class P1():
    """
    Store arrays to make the crystal mapping more
    efficient.

    Assume that Fourier space arrays are fft shifted.

    Perform symmetry operations with the np.fft.fftfreq basis
    so that (say) a flip operation behaves like:
    a         = [0, 1, 2, 3, 4, 5, 6, 7]
    a flipped = [0, 7, 6, 5, 4, 3, 2, 1]

    or:
    i         = np.fft.fftfreq(8)*8
              = [ 0,  1,  2,  3, -4, -3, -2, -1]
    i flipped = [ 0, -1, -2, -3, -4,  3,  2,  1]
    """
    def __init__(self, unitcell_size, det_shape, dtype=np.complex128):
        # store the tranlation ramps
        # x = x
        T0 = 1
        
        self.translations = np.array([T0])
        
        # keep an array for the 4 symmetry related coppies of the solid unit
        self.syms = np.zeros((1,) + tuple(det_shape), dtype=dtype)
    
    def solid_syms_Fourier(self, solid):
        """
        Take the Fourier space solid unit then return each
        of the symmetry related partners.
        """
        # x = x
        self.syms[0] = solid
        
        self.syms *= self.translations
        return self.syms


def lattice(unit_cell_size, shape):
    """
    We should just have delta functions at 1/d
    however, if the unit cell does not divide the 
    detector shape evenly then these fall between 
    pixels. 
    """
    # generate the q-space coordinates
    qi = np.fft.fftfreq(shape[0])
    qj = np.fft.fftfreq(shape[1])
    qk = np.fft.fftfreq(shape[2])
    
    # generate the recirocal lattice points from the unit cell size
    qs_unit = np.meshgrid(np.fft.fftfreq(unit_cell_size[0]), \
                          np.fft.fftfreq(unit_cell_size[1]), \
                          np.fft.fftfreq(unit_cell_size[2]), \
                          indexing = 'ij')
    
    # now we want qs[qs_unit] = 1.
    lattice = np.zeros(shape, dtype=np.float)
    for ii, jj, kk in zip(qs_unit[0].ravel(), qs_unit[1].ravel(), qs_unit[2].ravel()):
        i = np.argmin(np.abs(ii - qi))
        j = np.argmin(np.abs(jj - qj))
        k = np.argmin(np.abs(kk - qk))
        lattice[i, j, k] = 1.
    
    return lattice
