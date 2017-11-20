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
import sys, os
from itertools import product
from functools import reduce

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

#import symmetry_operations 
import padding
import add_noise_3d
import io_utils
import maps

import pyximport; pyximport.install()
#from ellipse_2D_cython import project_2D_Ellipse_arrays_cython
from ellipse_2D_cython_new import project_2D_Ellipse_arrays_cython_test

import phasing_3d
from phasing_3d.src.mappers import Modes
from phasing_3d.src.mappers import isValid

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

cuda_stream = drv.Stream()

import ellipse_2D_cuda


ell = ellipse_2D_cuda.gpu_fns.get_function("project_2D_Ellipse_arrays_cuda")

N = 128**3

r = lambda : np.random.random((128**3,))

#x = np.random.random((128**3,))
#y, Wx, Wy, I, u, v = x.copy(), x.copy(), x.copy(), x.copy(), x.copy(), x.copy()
"""
o = np.ones((N,), dtype=np.float64)
x = 0. * o.copy()
y = o.copy()
Wx = o.copy()
Wy = 0. * o.copy()
I = 4. * o.copy()
"""

x, y, Wx, Wy, I = r(), r(), r(), r(), r()

xp = -np.ones_like(x)
yp = -np.ones_like(x)
mask = np.ones_like(x).astype(np.uint8)

g = gpuarray.to_gpu
# to gpu
xpg = g(xp)
ypg = g(yp)

ell(np.uint32(len(x)), g(x), g(y), g(Wx), g(Wy), g(I), g(mask), xpg, ypg, block=(256,1,1), grid=(64,1))

print(xpg.get())
print(ypg.get())
