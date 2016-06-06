import numpy as np
from P1 import multiroll
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
        # only calculate the translations when they are needed
        self.translations = None
        
        self.unitcell_size = unitcell_size
        self.det_shape     = det_shape
        
        # keep an array for the 4 symmetry related coppies of the solid unit
        self.syms = np.zeros((4,) + tuple(det_shape), dtype=dtype)

    def make_Ts(self):
        det_shape     = self.det_shape
        unitcell_size = self.unitcell_size
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
    
    def solid_syms_Fourier(self, solid):
        self.syms = self.syms.astype(solid.dtype)
        
        # x = x
        self.syms[0] = solid
        
        # x = 0.5 + x, 0.5 - y, -z
        self.syms[1][:, 0, :]  = solid[:, 0, :]
        self.syms[1][:, 1:, :] = solid[:, -1:0:-1, :]
        self.syms[1][:, :, 1:] = self.syms[1][:, :, -1:0:-1]
        
        # x = -x, 0.5 + y, 0.5 - z
        self.syms[2][0, :, :]  = solid[0, :, :]
        self.syms[2][1:, :, :] = solid[-1:0:-1, :, :]
        self.syms[2][:, :, 1:] = self.syms[2][:, :, -1:0:-1]
        
        # x = 0.5 - x, -y, 0.5 + z
        self.syms[3][0, :, :]  = solid[0, :, :]
        self.syms[3][1:, :, :] = solid[-1:0:-1, :, :]
        self.syms[3][:, 1:, :] = self.syms[3][:, -1:0:-1, :]
        
        if self.translations is None :
            self.make_Ts()
        
        self.syms *= self.translations
        return self.syms

    def solid_syms_real(self, solid):
        """
        This uses pixel shifts (not phase ramps) for translation.
        Therefore sub-pixel shifts are ignored.
        """
        syms = self.syms #np.empty((4,) + solid.shape, dtype=solid.dtype)
        
        # x = x
        syms[0] = solid
        
        # x = 0.5 + x, 0.5 - y, -z
        syms[1][:, 0, :]  = solid[:, 0, :]
        syms[1][:, 1:, :] = solid[:, -1:0:-1, :]
        syms[1][:, :, 1:] = syms[1][:, :, -1:0:-1]
        
        # x = -x, 0.5 + y, 0.5 - z
        syms[2][0, :, :]  = solid[0, :, :]
        syms[2][1:, :, :] = solid[-1:0:-1, :, :]
        syms[2][:, :, 1:] = syms[2][:, :, -1:0:-1]
        
        # x = 0.5 - x, -y, 0.5 + z
        syms[3][0, :, :]  = solid[0, :, :]
        syms[3][1:, :, :] = solid[-1:0:-1, :, :]
        syms[3][:, 1:, :] = syms[3][:, -1:0:-1, :]
        
        translations = []
        translations.append([self.unitcell_size[0]/2, self.unitcell_size[1]/2, 0])
        translations.append([0, self.unitcell_size[1]/2, self.unitcell_size[2]/2])
        translations.append([self.unitcell_size[0]/2, 0, self.unitcell_size[2]/2])
        
        for i, t in enumerate(translations):
            syms[i+1] = multiroll(syms[i+1], t)
        return syms


def test_P212121():
    # make a unit cell
    unit_cell_size = tuple([8,8,4])
    solid_shape    = tuple([4,4,2])
    
    i = np.fft.fftfreq(unit_cell_size[0],1./float(unit_cell_size[0])).astype(np.int)
    j = np.fft.fftfreq(unit_cell_size[1],1./float(unit_cell_size[1])).astype(np.int)
    k = np.fft.fftfreq(unit_cell_size[2],1./float(unit_cell_size[2])).astype(np.int)
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    
    solid = np.random.random(solid_shape)
    Solid = np.fft.fftn(solid, unit_cell_size)
    solid = np.fft.ifftn(Solid)

    sym_ops   = P212121(unit_cell_size, unit_cell_size)
    unit_cell = sym_ops.solid_syms_Fourier(Solid)
    unit_cell = np.sum(unit_cell, axis=0)
    unit_cell = np.fft.ifftn(unit_cell)
    """
    # manual test :
    unit_cell_size = tuple([8,8,4])
    solid_shape    = tuple([4,4,2])
    
    solid = np.zeros(solid_shape)
    solid[1,2,1] = 1.

    # should become
    unit_cell = np.zeros_like(solid)
    unit_cell[1,2,1]   = 1.
    unit_cell[-3,2,-1] = 1.
    unit_cell[-1,-2,1] = 1.
    unit_cell[3,-2,-1] = 1.
    """

    # test symmetries
    u2 = np.array(unit_cell_size) / 2
    i1 = i
    j1 = j
    k1 = k
    print 'r1 = x, y, z             :', \
            np.allclose(unit_cell[i,j,k], unit_cell[i1,j1,k1])

    i2 =  ((i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0]
    j2 = ((-j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k2 = -k
    print 'r2 = 1/2 + x, 1/2 - y, -z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i2,j2,k2])

    i3 =  -i
    j3 = ((j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k3 = ((-k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print 'r3 = -x, 1/2 + y, 1/2 - z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i3,j3,k3])

    i4 = ((-i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0] 
    j4 = -j
    k4 = ((k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print 'r3 = 1/2 - x, -y, 1/2 + z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i4,j4,k4])

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


if __name__ == '__main__':
    # make a unit cell
    unit_cell_size = tuple([8,8,4])
    solid_shape    = tuple([4,4,2])
    
    i = np.fft.fftfreq(unit_cell_size[0],1./float(unit_cell_size[0])).astype(np.int)
    j = np.fft.fftfreq(unit_cell_size[1],1./float(unit_cell_size[1])).astype(np.int)
    k = np.fft.fftfreq(unit_cell_size[2],1./float(unit_cell_size[2])).astype(np.int)
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    
    solid = np.random.random(solid_shape)
    #solid = np.zeros(solid_shape)
    #solid[1,2,1] = 1.
    Solid = np.fft.fftn(solid, unit_cell_size)
    solid = np.fft.ifftn(Solid)

    sym_ops   = P212121(unit_cell_size, unit_cell_size)
    unit_cell = sym_ops.solid_syms_Fourier(Solid)
    unit_cell = np.sum(unit_cell, axis=0)
    unit_cell = np.fft.ifftn(unit_cell)
    """
    unit_cell = np.zeros_like(solid)
    unit_cell[1,2,1]   = 1.
    unit_cell[-3,2,-1] = 1.
    unit_cell[-1,-2,1] = 1.
    unit_cell[3,-2,-1] = 1.
    """

    # test symmetries
    u2 = np.array(unit_cell_size) / 2
    i1 = i
    j1 = j
    k1 = k
    print 'r1 = x, y, z             :', \
            np.allclose(unit_cell[i,j,k], unit_cell[i1,j1,k1])

    i2 =  ((i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0]
    j2 = ((-j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k2 = -k
    print 'r2 = 1/2 + x, 1/2 - y, -z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i2,j2,k2])

    i3 =  -i
    j3 = ((j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k3 = ((-k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print 'r3 = -x, 1/2 + y, 1/2 - z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i3,j3,k3])

    i4 = ((-i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0] 
    j4 = -j
    k4 = ((k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print 'r3 = 1/2 - x, -y, 1/2 + z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i4,j4,k4])
