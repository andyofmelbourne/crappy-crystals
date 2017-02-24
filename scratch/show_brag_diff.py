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

try :
    import ConfigParser as configparser 
except ImportError :
    import configparser 

import h5py
import numpy as np
import scipy.ndimage

import os, sys

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))
#sys.path.append(os.path.join(root, 'utils/phasing_3d/utils'))

import phasing_3d
from symmetry_operations import multiroll

merge = phasing_3d.utils.merge

# get the solid units 
solid_both = h5py.File('../hdf5/duck_both/duck_both.h5')['/phase/solid_unit'][()]
solid_diff = h5py.File('../hdf5/duck_diffuse/duck_diffuse.h5')['/phase/solid_unit'][()]
solid      = h5py.File('../hdf5/duck_diffuse/duck_diffuse.h5')['/forward_model/solid_unit'][()]
solid_brag = solid#h5py.File('../hdf5/duck_Bragg/duck_Bragg.h5')['/phase/solid_unit'][()]

# centre them
solid_both = multiroll(np.abs(merge.centre(solid_both)), [10,10,10])
solid_brag = multiroll(np.abs(merge.centre(solid_brag)), [10,10,10])
solid_diff = multiroll(np.abs(merge.centre(solid_diff)), [10,10,10])
solid      = multiroll(np.abs(merge.centre(solid)),      [10,10,10])

# roll and cut them down in size
solid_both = solid_both[:20,:20,:20]
solid_brag = solid_brag[:20,:20,:20]
solid_diff = solid_diff[:20,:20,:20]
solid      = solid[:20,:20,:20]

# join them together
solids = np.vstack((solid.transpose((0,1,2)), solid_brag.transpose((0,1,2)), solid_both.transpose((0,1,2)), solid_diff.transpose((0,1,2))))

from mayavi import mlab
cont = mlab.contour3d(100*solids, contours=8, opacity=0.7)
mlab.show()
