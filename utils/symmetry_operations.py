from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import math
from itertools import product
from functools import reduce


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
        self.unitcell_size = unitcell_size
        
        # keep an array for the 1 symmetry related coppies of the solid unit
        self.syms = np.zeros((1,) + tuple(det_shape), dtype=dtype)
    
    def solid_syms_Fourier(self, solid, apply_translation = True):
        """
        Take the Fourier space solid unit then return each
        of the symmetry related partners.
        """
        # x = x
        self.syms[0] = solid
        
        self.syms *= self.translations
        return self.syms

    def solid_syms_real(self, solid):
        # x = x
        self.syms[0] = solid
        return self.syms

    def unflip_modes_Fourier(self, U, apply_translation = True):
        out = np.empty_like(U)
        out[0] = U[0]
        return out
    
    def solid_syms_crystal_real(self, solid):
        """
        Generate the symmetry related copies of the real-space solid unit
        in the crystal. This includes all symmetry related coppies of the 
        solid unit that fit within the field-of-view (not just the unit-cell
        as in solid_syms_real).
        """
        # calculate the number of times that we need to tile the unit-cell 
        # unit so that we fill the field-of-view
        tiles = np.ceil(2*np.array(solid.shape, dtype=np.float) / np.array(self.unitcell_size, dtype=np.float) - 1.0).astype(np.int)
        
        # get the symmetry related coppies of solid in the unit-cell
        # hopefully these fit in the field-of-view...
        syms = self.solid_syms_real(solid)

        # un-fftshift them
        syms = np.fft.fftshift(syms, axes=(1,2,3))
        
        # now translate each solid sym by tiles in each dimension, this could really blow up...
        syms_crystal = np.zeros( (len(syms) * np.prod(tiles),) + syms.shape[1:], dtype=syms.dtype)
        syms_crystal[: len(syms)] = syms
        
        index = len(syms)
        for sym in syms :
            for i in (np.arange(tiles[0]) - tiles[0]//2):
                for j in (np.arange(tiles[1]) - tiles[1]//2):
                    for k in (np.arange(tiles[2]) - tiles[2]//2):
                        if i == 0 and j == 0 and k == 0 :
                            continue
                        shift = np.array([i, j, k]) * np.array(self.unitcell_size)
                        #print(i,j,k, shift, index)
                        syms_crystal[index] = multiroll_nowrap(sym, shift)
                        index += 1
        
        syms_crystal = np.fft.ifftshift(syms_crystal, axes=(1,2,3))
        return syms_crystal

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
        self.no_solid_units = 4
        
        self.unitcell_size = unitcell_size
        self.det_shape     = det_shape
        
        # keep an array for the 4 symmetry related coppies of the solid unit
        #self.syms = np.zeros((4,) + tuple(det_shape), dtype=dtype)

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
    
    def solid_syms_Fourier(self, solid, apply_translation = True, syms = None):
        if syms is None :
            syms = np.empty((4,) + solid.shape, dtype=solid.dtype) # syms 

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
        
        if apply_translation :
            if self.translations is None :
                self.make_Ts()
            
            syms *= self.translations
        return syms

    def solid_syms_Fourier_masked(self, solid, i, j, k, apply_translation = True, syms = None):
        """
        solid = full solid unit at the detector
        syms  = masked syms  
        """
        if syms is None :
            syms = np.empty((4,) + i.shape, dtype=solid.dtype) # syms 
        
        # x = x
        syms[0] = solid[(i,j,k)]
        
        # x = 0.5 + x, 0.5 - y, -z
        syms[1] = solid[(i, -j, -k)]
        
        # x = -x, 0.5 + y, 0.5 - z
        syms[2] = solid[(-i, j, -k)]
        
        # x = 0.5 - x, -y, 0.5 + z
        syms[3] = solid[(-i, -j, k)]
        
        if apply_translation :
            if self.translations is None :
                self.make_Ts()
            
            for ii in range(4):
                syms[ii] *= self.translations[ii][(i,j,k)]
        
        return syms

    def unflip_modes_Fourier(self, U, apply_translation=True, inplace = False):
        if inplace :
            U_inv = U
        else :
            U_inv = U.copy()
        
        if apply_translation :
            if self.translations is None :
                self.make_Ts()
            
            U_inv *= self.translations.conj()
        
        # x = x
        #U_inv[0] = U_inv[0]
        
        # x = 0.5 + x, 0.5 - y, -z
        #U_inv[1][:, 0, :]  = U_inv[1][:, 0, :]
        U_inv[1][:, 1:, :] = U_inv[1][:, -1:0:-1, :]
        U_inv[1][:, :, 1:] = U_inv[1][:, :, -1:0:-1]

        # x = -x, 0.5 + y, 0.5 - z
        #U_inv[2][0, :, :]  = U_inv[2][0, :, :]
        U_inv[2][1:, :, :] = U_inv[2][-1:0:-1, :, :]
        U_inv[2][:, :, 1:] = U_inv[2][:, :, -1:0:-1]
        
        # x = 0.5 - x, -y, 0.5 + z
        #U_inv[3][0, :, :]  = U_inv[3][0, :, :]
        U_inv[3][1:, :, :] = U_inv[3][-1:0:-1, :, :]
        U_inv[3][:, 1:, :] = U_inv[3][:, -1:0:-1, :]

        return U_inv

    def solid_syms_real(self, solid, syms=None):
        """
        This uses pixel shifts (not phase ramps) for translation.
        Therefore sub-pixel shifts are ignored.
        """
        if syms is None :
            syms = np.empty((4,) + solid.shape, dtype=solid.dtype) # self.syms 
        
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
        translations.append([self.unitcell_size[0]//2, self.unitcell_size[1]//2, 0])
        translations.append([0, self.unitcell_size[1]//2, self.unitcell_size[2]//2])
        translations.append([self.unitcell_size[0]//2, 0, self.unitcell_size[2]//2])
        
        for i, t in enumerate(translations):
            syms[i+1] = multiroll(syms[i+1], t)
        return syms

    def solid_to_crystal_real(self, solid, return_unit=False):
        """
        Generate the symmetry related copies of the real-space solid unit
        in the crystal. This includes all symmetry related coppies of the 
        solid unit that fit within the field-of-view (not just the unit-cell
        as in solid_syms_real).
        """
        # calculate the number of times that we need to tile the unit-cell 
        # unit so that we fill the field-of-view
        tiles = np.ceil(2*np.array(solid.shape, dtype=np.float) / np.array(self.unitcell_size, dtype=np.float) - 1.0).astype(np.int)
        
        # get the symmetry related coppies of solid in the unit-cell
        # hopefully these fit in the field-of-view...
        U = np.sum(self.solid_syms_real(solid), axis=0)
        
        # un-fftshift them
        U = np.fft.fftshift(U)
        
        # now translate the unit-cell by tiles in each dimension
        C = U.copy()

        # temp array to speed up multiroll_nowrap
        t = np.asarray(np.zeros_like(U))
        
        for i in (np.arange(tiles[0]) - tiles[0]//2):
            for j in (np.arange(tiles[1]) - tiles[1]//2):
                for k in (np.arange(tiles[2]) - tiles[2]//2):
                    if i == 0 and j == 0 and k == 0 :
                        continue
                    
                    shift = np.array([i, j, k]) * np.array(self.unitcell_size)
                    
                    #print(i,j,k, shift, U.shape)
                    #print(i,j,k, shift, index)
                    C += multiroll_nowrap(U, shift, y = t)
        
        # re-fftshift them
        C = np.fft.ifftshift(C)
        
        if return_unit :
            return C, np.fft.ifftshift(U)
        else :
            return C


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
    u2 = np.array(unit_cell_size) // 2
    i1 = i
    j1 = j
    k1 = k
    print('r1 = x, y, z             :', \
            np.allclose(unit_cell[i,j,k], unit_cell[i1,j1,k1]))

    i2 =  ((i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0]
    j2 = ((-j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k2 = -k
    print('r2 = 1/2 + x, 1/2 - y, -z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i2,j2,k2]))

    i3 =  -i
    j3 = ((j + u2[1] + u2[1]) % unit_cell_size[1]) - u2[1] 
    k3 = ((-k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print('r3 = -x, 1/2 + y, 1/2 - z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i3,j3,k3]))

    i4 = ((-i + u2[0] + u2[0]) % unit_cell_size[0]) - u2[0] 
    j4 = -j
    k4 = ((k + u2[2] + u2[2]) % unit_cell_size[2]) - u2[2] 
    print('r3 = 1/2 - x, -y, 1/2 + z:', \
            np.allclose(unit_cell[i,j,k], unit_cell[i4,j4,k4]))

def T_fourier(shape, T, is_fft_shifted = True):
    """
    e - 2pi i r q
    e - 2pi i dx n m / N dx
    e - 2pi i n m / N 
    """
    # make i, j, k for each pixel
    if T[0] != 0 :
        i = np.fft.fftfreq(shape[0]) 
        i = np.exp(- 2J * np.pi * i * T[0])
    else :
        i = np.ones((shape[0],), dtype=np.float)
    
    if T[1] != 0 :
        j = np.fft.fftfreq(shape[1])
        j = np.exp(- 2J * np.pi * j * T[1])
    else :
        j = np.ones((shape[1],), dtype=np.float)
    
    if T[2] != 0 :
        k = np.fft.fftfreq(shape[2])
        k = np.exp(- 2J * np.pi * k * T[2])
    else :
        k = np.ones((shape[2],), dtype=np.float)
    
    phase_ramp = reduce(np.multiply.outer, [i, j, k])
    
    if is_fft_shifted is False :
        phase_ramp = np.fft.ifftshift(phase_ramp)
           
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

def multiroll(x, shift, axis=None):
    """Roll an array along each axis.

    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.

    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.

    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.

    See Also
    --------
    numpy.roll

    Example
    -------
    Here's a two-dimensional array:

    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    Roll the first axis one step and the second axis three steps:

    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    That's equivalent to:

    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:

    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    which is equivalent to:

    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y

def multiroll_nowrap(x, shift, axis=None, y = None):
    """Roll an array along each axis.

    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.

    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.

    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.

    See Also
    --------
    numpy.roll

    Example
    -------
    Here's a two-dimensional array:

    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    Roll the first axis one step and the second axis three steps:

    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    That's equivalent to:

    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:

    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    which is equivalent to:

    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    #shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    if y is None :
        y = np.zeros_like(x)
    else :
        y.fill(0)
    src_slices = [(slice(-shft, n),) if shft < 0 else (slice(0, n-shft),)
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, n+shft),) if shft < 0 else (slice(shft, n),)
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y

def lattice_old(unit_cell_size, shape):
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

def lattice(unit_cell_size, shape):
    """
    We should just have delta functions at 1/d
    however, if the unit cell does not divide the 
    detector shape evenly then these fall between 
    pixels. 
    """
    lattice = np.zeros(shape, dtype=np.float)

    # generate the q-space coordinates
    qi = np.fft.fftfreq(shape[0])
    qj = np.fft.fftfreq(shape[1])
    qk = np.fft.fftfreq(shape[2])
    
    ui = np.fft.fftfreq(unit_cell_size[0])
    uj = np.fft.fftfreq(unit_cell_size[1])
    uk = np.fft.fftfreq(unit_cell_size[2])

    i = np.zeros_like(qi)
    j = np.zeros_like(qj)
    k = np.zeros_like(qk)
    
    for ii in ui :
        i[np.argmin(np.abs(ii - qi))] = 1.
    for jj in uj :
        j[np.argmin(np.abs(jj - qj))] = 1.
    for kk in uk :
        k[np.argmin(np.abs(kk - qk))] = 1.

    lattice = reduce(np.multiply.outer, [i, j, k])
    
    #print(np.allclose(lattice, lattice_old(unit_cell_size, shape)))
    
    return lattice
