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
import os

def interp_3d(array, shapeout):
    from scipy.interpolate import griddata
    ijk = np.indices(array.shape)
    
    points = np.zeros((array.size, 3), dtype=np.float)
    points[:, 0] = ijk[0].ravel()
    points[:, 1] = ijk[1].ravel()
    points[:, 2] = ijk[2].ravel()
    values = array.astype(np.float).ravel()

    gridout  = np.mgrid[0: array.shape[0]-1: shapeout[0]*1j, \
                        0: array.shape[1]-1: shapeout[1]*1j, \
                        0: array.shape[2]-1: shapeout[2]*1j]
    arrayout = griddata(points, values, (gridout[0], gridout[1], gridout[2]), method='nearest')
    return arrayout
    
def interp_2d(array, shapeout):
    from scipy.interpolate import griddata
    ijk = np.indices(array.shape)
    
    points = np.zeros((array.size, 2), dtype=np.float)
    points[:, 0] = ijk[0].ravel()
    points[:, 1] = ijk[1].ravel()
    values = array.astype(np.float).ravel()

    gridout  = np.mgrid[0: array.shape[0]-1: shapeout[0]*1j, \
                        0: array.shape[1]-1: shapeout[1]*1j]
    arrayout = griddata(points, values, (gridout[0], gridout[1]), method='nearest')
    return arrayout

def make_3D_duck(shape = (12, 25, 30)):
    script_dir = os.path.dirname(__file__)
    duck_fnam  = os.path.join(script_dir, 'duck_300_211_8bit.raw')
    
    # call in a low res 2d duck image
    duck = np.fromfile(duck_fnam, dtype=np.int8).reshape((211, 300))
    
    # convert to bool
    duck = duck < 50

    # interpolate onto the desired grid
    duck = interp_2d(duck, shape[1:])

    # make a 3d volume
    duck3d = np.zeros(shape , dtype=np.bool)

    # loop over the first dimension with an expanding circle
    i, j = np.mgrid[0 :shape[1], 0 :shape[2]]
    
    origin = [int(1.5*shape[1]/2), shape[2]//2]

    r = np.sqrt( ((i-origin[0])**2 + (j-origin[1])**2).astype(np.float) )

    rs = list(range(shape[0]//2)) + list(range(shape[0]//2, 0, -1))
    rs = np.array(rs) * min(shape[1], shape[2]) / (shape[0]/2.)
    
    circle = lambda ri : r < ri
    
    for z in range(duck3d.shape[0]):
        duck3d[z, :, :] = circle(rs[z]) * duck
    
    # zero the edges
    duck3d[0, :, :] = 0
    duck3d[:, 0, :] = 0
    duck3d[:, :, 0] = 0
    return duck3d

        
if __name__ == '__main__':
    #duck3d = make_3D_duck()
    duck3d = make_3D_duck2()
