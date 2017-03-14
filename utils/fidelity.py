#!/usr/bin/env python

# for python 2 / 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try :
    range = xrange
except NameError :
    pass

import numpy as np
from symmetry_operations import multiroll

def centre(O):
    import scipy.ndimage
    a  = (O * O.conj()).real
    a  = np.fft.fftshift(a)

    aroll = []
    for i in range(len(a.shape)):
        axes = list(range(len(a.shape)))
        axes.pop(i)
        t = np.sum(a, axis = tuple(axes))
        
        dcm = [scipy.ndimage.measurements.center_of_mass(np.roll(t, i+1))[0] - \
               scipy.ndimage.measurements.center_of_mass(np.roll(t, i  ))[0]   \
               for i in range(t.shape[0])]
        
        dcm = scipy.ndimage.gaussian_filter1d(dcm, t.shape[0]/3., mode='wrap')
        
        aroll.append(np.argmax(dcm))
    
    
    # roughly centre O
    O = multiroll(O, aroll)
    O = np.fft.fftshift(O)

    cm = np.rint(scipy.ndimage.measurements.center_of_mass( (O*O.conj()).real)).astype(np.int)
    O  = multiroll(O, -cm)
    
    # roughly centre O
    #O = multiroll(O, aroll)
    
    # this doesn't really work (lot's of smearing from the fourier interpolation)
    # fourier shift to the centre of mass
    #O = np.fft.ifftshift(O)
    #cm = scipy.ndimage.measurements.center_of_mass((O * O.conj()).real)
    #print cm, aroll
    #O  = roll(O, cm)
    return O

def _calc_fid(o_known, o):
    # allign phases
    s     = np.sum(o_known)
    phase = np.arctan2(s.imag, s.real)
    o_known = o_known * np.exp(- 1J * phase)

    s     = np.sum(o)
    phase = np.arctan2(s.imag, s.real)
    o     = o * np.exp(- 1J * phase)

    # flip / unflip
    er1 = np.sum( np.abs( o_known - o)**2 )
    o[1:, :, :] = o[-1:0:-1, :, :]
    o[:, 1:, :] = o[:, -1:0:-1, :]
    o[:, :, 1:] = o[:, :, -1:0:-1]
    er2 = np.sum( np.abs( o_known - o)**2 )
    fid = min(er1, er2) / np.sum(np.abs(o_known)**2)
    return fid

def calculate_fidelity(o_known, o):
    """
    calculate:
        fid = min{ sum |o_known - o(+-r) e^{i phi)|^2 / sum |o_known|^2 }
        over phi and +-

        fid_trans = min{ sum |o_known - o(+-r - r') e^{i phi)|^2 / sum |o_known|^2 }
        over phi, +-, and r'
    """
    fid = _calc_fid(o_known, o)
    
    # with centering
    fid_trans = _calc_fid(centre(o_known), centre(o))
    return fid, fid_trans
