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

import numpy as np
import h5py 
import argparse
import os, sys
import shlex
from subprocess import PIPE, Popen
from scipy.ndimage.filters import gaussian_filter

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

from calculate_constraint_ratio import calculate_constraint_ratio
import io_utils
import forward_sim
import ccp4_reader
import pdb_parser

def mayavi_plot(map, geom, atom_coor=None, show=True):
    '''
    plots 3D map data and atom coordinates from pdb (optional)\n
    geom : dictionary {} containing geometry parameters \n
    - 'originx' = [x,y,z] coordinates of map origin relative to UC origin (Angstroms) \n
    - 'vox' = Voxel dimensions (Angstroms) \n
    - 'abc' = UC lenths (Angstroms) \n
    atom_coor (optional): if given, atom coordinates from pdb are ploted \n
    show : Set it to False if another plot follows which you want to plot simultaniously \n
            if True, malab.show() is called --> Start interacting with figure \n
            if False, will just create a new figure object w/o halting the program
    '''
    from mayavi import mlab
    vox = geom['vox']
    # make x y z grids
    xx,yy,zz = np.mgrid[ 0:(map.shape[0]*vox[0]-0.0001):vox[0], \
                0:(map.shape[1]*vox[1]-0.0001):vox[1], 0:(map.shape[2]*vox[2]-0.0001):vox[2] ]
    mean = np.mean(map[np.nonzero(map)])
    sigma = np.std(map[np.nonzero(map)])
    mlab.contour3d(xx, yy, zz, map, contours=[mean-2*sigma,mean-sigma,mean,mean+sigma,mean+2*sigma], \
                   transparent=True, opacity=0.5, vmin=(mean-2*sigma), vmax=(mean+2*sigma))
    mlab.outline()
    mlab.orientation_axes()
    mlab.scalarbar(orientation='vertical')
    mlab.axes(nb_labels=10, xlabel='x (A)', ylabel='y (A)', x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=False)
    if atom_coor is not None:
        originx = geom['originx']
        x = (atom_coor[0] - originx[0]) % (geom['vox'][0]*map.shape[0])
        y = (atom_coor[1] - originx[1]) % (geom['vox'][1]*map.shape[1])
        z = (atom_coor[2] - originx[2]) % (geom['vox'][2]*map.shape[2])
        mlab.points3d(x, y, z, scale_factor=0.5, color=(1.0,1.0,1.0))
    if show:
        mlab.show()
    else:
        mlab.figure() 

def parse_cmdline_args(default_config='show_contour.ini'):
    parser = argparse.ArgumentParser(description="Use mayavi to display a contour plot of some 3D data.")
    parser.add_argument('-f', '--filename', type=str, \
                        help="file name of the *.h5 file to read")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # if config is None then read the default from the *.h5 dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], default_config)
        if not os.path.exists(args.config):
            args.config = '../process/' + default_config
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = io_utils.parse_parameters(config)[default_config[:-4]]
    
    return args, params

if __name__ == '__main__':
    #args, params = parse_cmdline_args()
    
    # now 
    #geom = {'vox': [1.,1.,1.], 'abc': [50.,50.,50.]}
    
    f = h5py.File('hdf5/pdb/pdb.h5')
    map = f['forward_model_pdb/unit_cell'][()].real
    map = np.fft.fftshift(map)
    vox = np.array([4.0,4.0,4.0])
    originx = -np.array(map.shape)//2 * vox
    geom = {'vox': vox, 'originx' : originx}

    xyz = pdb_parser.coor('temp/2AXT.pdb')
    # wrap them
    #xyz[0] = xyz[0] % (geom['vox'][0]*map.shape[0])
    #xyz[1] = xyz[1] % (geom['vox'][1]*map.shape[1])
    #xyz[2] = xyz[2] % (geom['vox'][2]*map.shape[2])
    
    # generate symmetry partners
    abc = f['forward_model_pdb/unit_cell_ang'][()]
    # x = 0.5 + x, 0.5 - y, -z
    xyz2 = np.empty_like(xyz)
    xyz2[0] = abc[0]/2. + xyz[0]
    xyz2[1] = abc[1]/2. - xyz[1]
    xyz2[2] = - xyz[2]
    # x = -x, 0.5 + y, 0.5 - z
    xyz3 = np.empty_like(xyz)
    xyz3[0] = - xyz[0]
    xyz3[1] = abc[1]/2. + xyz[1]
    xyz3[2] = abc[2]/2. - xyz[2]
    # x = 0.5 - x, -y, 0.5 + z
    xyz4 = np.empty_like(xyz)
    xyz4[0] = abc[0]/2. - xyz[0]
    xyz4[1] = - xyz[1]
    xyz4[2] = abc[2]/2. + xyz[2]

    xyz = np.hstack((xyz, xyz2, xyz3, xyz4))

    print(xyz.shape)
    mayavi_plot(map, geom, atom_coor=xyz, show=True)

