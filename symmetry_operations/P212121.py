import numpy as np
from P1 import lattice

name = 'P212121'

class P212121():
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
        T0 = np.ones(det_shape, dtype=np.complex128)
        # x = 0.5 + x, 0.5 - y, -z
        T1 = T_fourier(det_shape, [unitcell_size[0]/2., unitcell_size[1]/2., 0.0])
        # x = -x, 0.5 + y, 0.5 - z
        T2 = T_fourier(det_shape, [0.0, unitcell_size[1]/2., unitcell_size[2]/2.])
        # x = 0.5 - x, -y, 0.5 + z
        T3 = T_fourier(det_shape, [unitcell_size[0]/2., 0.0, unitcell_size[2]/2.])
        
        self.translations = np.array([T0, T1, T2, T3])
        
        # keep an array for the 4 symmetry related coppies of the solid unit
        self.flips = np.zeros((4,) + det_shape, dtype=dtype)
    
    def solid_syms_Fourier(self, solid):
        # x = x
        self.flips[0] = solid
        # x = 0.5 + x, 0.5 - y, -z
        self.flips[1] = solid[:, ::-1, ::-1]


def T_fourier(shape, T, is_fft_shifted = True):
    """
    e - 2pi i r q
    e - 2pi i dx n m / N dx
    e - 2pi i n m / N 
    """
    # make i, j, k for each pixel
    i = np.fft.fftfreq(shape[0]) 
    j = np.fft.fftfreq(shape[1])
    k = np.fft.fftfreq(shape[2])
    i, j, k = np.meshgrid(i, j, k, indexing='ij')

    if is_fft_shifted is False :
        i = np.fft.ifftshift(i)
        j = np.fft.ifftshift(j)
        k = np.fft.ifftshift(k)

    phase_ramp = np.exp(- 2J * np.pi * (i * T[0] + j * T[1] + k * T[2]))
    return phase_ramp


def solid_syms(solid_unit, unitcell_size, det_shape):
    """
    Take the solid unit and map it 
    to the detector. Then return each
    of the symmetry related partners.
    """
    modes = []
    unitcell = solid_unit #np.fft.fftn(solid_unit, det_shape)
    modes.append(unitcell.copy())
    
    # x = 0.5 + x, 0.5 - y, -z
    temp  = unitcell[::, ::-1, ::-1].copy()
    temp *= T_fourier(temp.shape, [unitcell_size[0]/2., unitcell_size[1]/2., 0.0])
    modes.append(temp.copy())
    
    # x = -x, 0.5 + y, 0.5 - z
    temp  = unitcell[::-1, ::, ::-1].copy()
    temp *= T_fourier(temp.shape, [0.0, unitcell_size[1]/2., unitcell_size[2]/2.])
    modes.append(temp.copy())

    # x = 0.5 - x, -y, 0.5 + z
    temp  = unitcell[::-1, ::-1, ::].copy()
    temp *= T_fourier(temp.shape, [unitcell_size[0]/2., 0.0, unitcell_size[2]/2.])
    modes.append(temp.copy())
    return np.array(modes)


def unit_cell(solid_unit, unitcell_size):
    """
    see p212121diagramme.gif

    u = sum_i o(Ri . r + T)

    in Fourier space 
    U = sum_i O(Ri . q) * e^{-2pi i T . q)

    what about a lookup table for the Ri?
    
    x = y = z = 0 is on the courner of the origin pixel
    """
    array = np.zeros(unitcell_size, dtype=np.float) 
    array[: solid_unit.shape[0], : solid_unit.shape[1], : solid_unit.shape[2]] = solid_unit

    array = solid_syms(array, array.shape, array.shape)
    array = np.fft.ifftn(np.sum(array, axis=0))
    return array

