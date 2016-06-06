import numpy as np
from itertools import product

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

    def solid_syms_real(self, solid):
        # x = x
        self.syms[0] = solid
        return self.syms


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
