import h5py
import numpy as np
import os

fnam = '../hdf5/5jdk/5jdk.h5'
dnam = 'process_3/gaus'
sigma = 5.

if not os.path.exists(fnam):
    import sys
    print('no such file:', fnam)
    sys.exit()

f = h5py.File(fnam)
shape = f['forward_model/solid_unit'].shape

dxyz = np.array([0.575625, 0.63875, 1.0959375])
i, j, k = [np.fft.fftfreq(shape[i], 1./float(shape[i]))*dxyz[i] for i in range(3)]
print(i)
i, j, k = np.meshgrid(i, j, k, indexing='ij')

gaus = np.exp( - ((i+25.)**2 + (j-5.)**2 + (k-10.)**2) / (2. * sigma**2))


if dnam in f :
    del f[dnam]

f[dnam] = gaus
f.close()

