import numpy as np
from P1 import lattice

name = 'P212121'

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
    unitcell = np.fft.fftn(solid_unit, det_shape)
    modes.append(unitcell)
    
    # x = 0.5 + x, 0.5 - y, -z
    temp  = unitcell[::, ::-1, ::-1].copy()
    temp *= T_fourier(temp.shape, [unitcell_size[0]/2., unitcell_size[1]/2., 0.0])
    modes.append(temp)
    
    # x = -x, 0.5 + y, 0.5 - z
    temp  = unitcell[::-1, ::, ::-1].copy()
    temp *= T_fourier(temp.shape, [0.0, unitcell_size[1]/2., unitcell_size[2]/2.])
    modes.append(temp)

    # x = 0.5 - x, -y, 0.5 + z
    temp  = unitcell[::-1, ::-1, ::].copy()
    temp *= T_fourier(temp.shape, [unitcell_size[0]/2., 0.0, unitcell_size[2]/2.])
    modes.append(temp)
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

