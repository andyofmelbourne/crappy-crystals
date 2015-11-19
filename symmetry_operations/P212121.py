import numpy as np

name = 'P212121'

def solid_syms(solid_unit, unitcell_size, det_shape):
    """
    Take the solid unit and map it 
    to the detector. Then return each
    of the symmetry related partners.
    """
    #unitcell = unit_cell(solid_unit, unitcell_size)
    #unitcell = np.fft.fftn(unitcell, det_shape)
    unitcell = np.fft.fftn(solid_unit, det_shape)
    return unitcell[np.newaxis, :, :, :]

def unit_cell(solid_unit, unitcell_size):
    array = np.zeros(unitcell_size, dtype=np.float) 
    array[: solid_unit.shape[0], : solid_unit.shape[1], : solid_unit.shape[2]] = solid_unit
    return array

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
