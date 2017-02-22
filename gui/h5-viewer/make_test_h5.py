#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import numpy as np


print('writing to test.h5...')
f = h5py.File('test.h5')

a3d = np.random.random((10,256,256))
a2d = np.random.random((1024,512))
a1d = np.arange(1000)

a2d_com = np.random.random((1024,512)) + 1J * np.random.random((1024,512))

if '/3D_array' in f:
    del f['/3D_array'] 
f['3D_array'] = a3d

if '/2D_array' in f:
    del f['/2D_array'] 
f['2D_array'] = a2d

if '/2D_array_complex' in f:
    del f['/2D_array_complex'] 
f['2D_array_complex'] = a2d_com

if '/1D_array' in f:
    del f['/1D_array'] 
f['1D_array'] = a1d

if '/text' in f:
    del f['/text'] 
f['text'] = 'hello world!'

if '/number' in f:
    del f['/number'] 
f['number'] = 3.142

f.close()
print('Done!')
print('\nNow run the viewer with:')
print('python h5_viewer.py test.h5')


