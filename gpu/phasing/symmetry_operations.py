import numpy as np
import afnumpy
from itertools import product

import crappy_crystals
from crappy_crystals.phasing.symmetry_operations import *

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
        
        self.translations = afnumpy.array([T0])
        
        # keep an array for the 1 symmetry related coppies of the solid unit
        self.syms = afnumpy.zeros((1,) + tuple(det_shape), dtype=dtype)
    
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
        self.syms = afnumpy.zeros((4,) + tuple(det_shape), dtype=dtype)

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
        self.translations = afnumpy.array([T0, T1, T2, T3])
    
    def solid_syms_Fourier(self, solid):
        # x = 0.5 + x, 0.5 - y, -z
        u1 = afnumpy.arrayfire.data.flip(solid.d_array, dim=1)
        u1 = afnumpy.arrayfire.data.flip(u1, dim=0)
        u1 = afnumpy.arrayfire.data.shift(u1, 1, d1=1)

        # x = -x, 0.5 + y, 0.5 - z
        u2 = afnumpy.arrayfire.data.flip(solid.d_array, dim=2)
        u2 = afnumpy.arrayfire.data.flip(u2, dim=0)
        u2 = afnumpy.arrayfire.data.shift(u2, 1, d2=1)

        # x = 0.5 - x, -y, 0.5 + z
        u3 = afnumpy.arrayfire.data.flip(solid.d_array, dim=2)
        u3 = afnumpy.arrayfire.data.flip(u3, dim=1)
        u3 = afnumpy.arrayfire.data.shift(u3, 0, d1=1, d2=1)

        U = afnumpy.arrayfire.data.join(3, solid.d_array, u1, u2, u3)
        U = afnumpy.arrayfire.data.moddims(U, self.syms.shape[3],self.syms.shape[2],self.syms.shape[1],self.syms.shape[0])
        self.syms = afnumpy.asarray(U)
        
        if self.translations is None :
            self.make_Ts()
        
        self.syms *= self.translations
        #return np.array([np.array(afnumpy.array(i)) for i in [solid, u1, u2, u3]]) # afnumpy.array(U) #self.syms
        #return afnumpy.array(U) * self.translations #self.syms
        return self.syms

    def solid_syms_real(self, solid):
        """
        This uses pixel shifts (not phase ramps) for translation.
        Therefore sub-pixel shifts are ignored.
        """
        translations = []
        translations.append([self.unitcell_size[0]/2, self.unitcell_size[1]/2, 0])
        translations.append([0, self.unitcell_size[1]/2, self.unitcell_size[2]/2])
        translations.append([self.unitcell_size[0]/2, 0, self.unitcell_size[2]/2])
        
        # x = 0.5 + x, 0.5 - y, -z
        u1 = afnumpy.arrayfire.data.flip(solid.d_array, dim=1)
        u1 = afnumpy.arrayfire.data.flip(u1, dim=0)
        u1 = afnumpy.arrayfire.data.shift(u1, 1, d1=1)
        
        t  = translations[0]
        u1 = afnumpy.arrayfire.data.shift(u1, t[-1], d1=t[-2], d2=t[-3])

        # x = -x, 0.5 + y, 0.5 - z
        u2 = afnumpy.arrayfire.data.flip(solid.d_array, dim=2)
        u2 = afnumpy.arrayfire.data.flip(u2, dim=0)
        u2 = afnumpy.arrayfire.data.shift(u2, 1, d2=1)

        t  = translations[1]
        u2 = afnumpy.arrayfire.data.shift(u2, t[-1], d1=t[-2], d2=t[-3])

        # x = 0.5 - x, -y, 0.5 + z
        u3 = afnumpy.arrayfire.data.flip(solid.d_array, dim=2)
        u3 = afnumpy.arrayfire.data.flip(u3, dim=1)
        u3 = afnumpy.arrayfire.data.shift(u3, 0, d1=1, d2=1)
        
        t  = translations[2]
        u3 = afnumpy.arrayfire.data.shift(u3, t[-1], d1=t[-2], d2=t[-3])

        U = afnumpy.arrayfire.data.join(3, solid.d_array, u1, u2, u3)
        U = afnumpy.arrayfire.data.moddims(U, self.syms.shape[3],self.syms.shape[2],self.syms.shape[1],self.syms.shape[0])
        U = afnumpy.asarray(U)
        
        return U
