import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import numpy as np
import scipy.ndimage

import os, sys
# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import io_utils
import duck_3D


f = h5py.File('cheshire_scan.h5')
error_map = f['error_map'][:, :, 0].T
crystal   = f['crystal'][()]
crystal   = np.sum(crystal.real, axis=2).T
#crystal   = duck_3D.interp_2d(np.sum(crystal.real, axis=2), (256, 256))
#crystal   = np.fft.fftshift(crystal)[:, :] #[64 : 256-64, 64 : 256-64]

# smooth it 
emin, emax = error_map.min(), error_map.max()
error_map = duck_3D.interp_2d(error_map, (256, 256))
error_map = scipy.ndimage.filters.gaussian_filter(error_map, 4.0)
error_map -= error_map.min()
error_map *= (emax-emin) / error_map.max() 
error_map += emin

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(1, 1) #, width_ratios=[1, 1])
ax = plt.subplot(gs[0])
im = ax.imshow(error_map, cmap = plt.cm.hot, alpha = 0.7, origin='lower left')
ax.set_xlabel('$\Delta a$', fontsize = 16)
ax.set_xticks([0, 128, 255])
ax.set_xticklabels(['0', '1/2', '1'])
ax.set_ylabel('$\Delta b$', fontsize = 16)
ax.set_yticks([0, 128, 255])
ax.set_yticklabels(['0', '1/2', '1'])
#ax.set_yticks([0, 0.5, 1.])
#ax.contour(error_map)
ax.set_title(r'$\sum |\hat{P}_{\rm{data}} \cdot \Psi - \Psi |^2$')

# adding the Contour lines with labels
cset = ax.contour(error_map, linewidths=2, cmap = plt.cm.gray)
ax.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

# add a box showing the cheshire cell
ax.hlines(129, 0, 129, colors='k', linestyles='dashed', linewidth=2)
ax.vlines(129, 0, 129, colors='k', linestyles='dashed', linewidth=2)
"""
ax2 = plt.subplot(gs[1])
im2 = ax2.imshow(crystal, cmap='Greys', origin='lower left')
ax2.set_xlabel('$a$')
ax2.set_xticks([0, 32, 64])
ax2.set_xticklabels(['0', '1', '2'])
ax2.set_ylabel('$b$')
ax2.set_yticks([0, 32, 64])
ax2.set_yticklabels(['0', '1', '2'])
"""

plt.colorbar(im)
#plt.show()
plt.savefig('cheshire_scan_110.svg')
