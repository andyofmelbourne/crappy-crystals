#!/usr/bin/env python
"""
Get the crystal info:
    (phenix.fetch_pdb)
    pdbid --> pdbid.pdb, pdbid-sf.cif 

Generate the cut-map for the solid-unit:
    (phenix.maps)
    pdbid.pdb, pdbid-sf.cif --> pdbid-map.ccp4

Cut out the solid-unit from the map using atomic coords
from the pdb
    (get_mol_density_ccp4_pdb)
    cut-map --> cut_density

Place in a bigger array correctly with respect to the origin.
Then interpolate onto the desired grid.

Then generate the forward map for the diffraction.
"""

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

def parse_cmdline_args(default_config='forward_model_pdb.ini'):
    parser = argparse.ArgumentParser(description="calculate the forward model diffraction intensity for a disorded crystal who's structure is given by a pdb entry. The results are output into a .h5 file.")
    parser.add_argument('-f', '--filename', type=str, \
                        help="file name of the *.h5 file to edit / create")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # if config is non then read the default from the *.h5 dir
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

def make_maps_params(pdb_fnam, cif_fnam, output, res_ang = 0.25):
    outputdir = os.path.split(output)[0]
    args = {'pdb_fnam': pdb_fnam, 'cif_fnam': cif_fnam, 'outputdir': outputdir, 'output': output, 'res': res_ang}
    maps_params =  \
    """
     maps {{
       input {{
         pdb_file_name = "{pdb_fnam}"
         reflection_data {{
           file_name = "{cif_fnam}"
           labels = None
           high_resolution = None
           low_resolution = None
           outliers_rejection = True
           french_wilson_scale = True
           french_wilson {{
             max_bins = 60
             min_bin_size = 40
           }}
           sigma_fobs_rejection_criterion = None
           sigma_iobs_rejection_criterion = None
           r_free_flags {{
             file_name = None
             label = None
             test_flag_value = None
             ignore_r_free_flags = False
           }}
         }}
       }}
       output {{
         directory = "{outputdir}"
         prefix = None
         job_title = None
         fmodel_data_file_format = mtz
         include_r_free_flags = False
       }}
       scattering_table = wk1995 it1992 *n_gaussian neutron electron
       wavelength = None
       bulk_solvent_correction = True
       anisotropic_scaling = True
       skip_twin_detection = False
       omit {{
         method = *simple
         selection = None
       }}
       map {{
         map_type = DFc
         format = xplor *ccp4
         file_name = "{output}"
         fill_missing_f_obs = False
         grid_resolution_factor = {res}
         scale = *sigma volume
         region = *selection cell
         atom_selection = None
         atom_selection_buffer = 3
         acentrics_scale = 2
         centrics_pre_scale = 1
         sharpening = False
         sharpening_b_factor = None
         exclude_free_r_reflections = False
         isotropize = True
       }}
     }}
    """.format(**args)
    return maps_params

def get_pdb_mtz(pdbid, dirnam = './temp'):
    # fetch the pdb info
    from subprocess import PIPE, Popen
    import shlex
    import os
    #dirnam2 = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), dirnam))
    dirnam2 = os.path.abspath(dirnam)
    
    cmd = 'mkdir ' + dirnam2
    p = Popen(shlex.split(cmd))
    p.wait()
    
    cmd = 'phenix.fetch_pdb --mtz ' + pdbid
    print(cmd, dirnam2)
    p = Popen(shlex.split(cmd), cwd=dirnam2)
    p.wait()

def make_map_ccp4(pdbid):
    # make the maps.params file
    dirnam = os.path.abspath('./temp')
    pdb_fnam = os.path.abspath(os.path.join(dirnam, pdbid + '.pdb'))
    cif_fnam = os.path.abspath(os.path.join(dirnam, pdbid + '-sf.cif'))
    output   = os.path.abspath(os.path.join(dirnam, pdbid + '-map.ccp4'))
    maps_params = make_maps_params(pdb_fnam, cif_fnam, output)
    
    # if the output file already exists then skip this stuff
    if os.path.exists(output):
        return output, pdb_fnam
    
    # if the .pdb file is not present then get it
    if not os.path.exists(pdb_fnam):
        get_pdb_mtz(pdbid, dirnam)
    
    # if the file exists then delete it 
    fnam = os.path.abspath(os.path.join(dirnam, 'maps.params'))
    if os.path.exists(fnam):
        os.remove(fnam)
    
    f = open(fnam, 'w')
    f.writelines(maps_params)
    f.close()
    
    # run the code 
    cmd = 'phenix.maps ' + fnam
    p = Popen(shlex.split(cmd))
    p.wait()
    return output, pdb_fnam

def get_origin_voxel_unit(cdata):
    """
    cdata is the ccp4 file data as returned by ccp4_reader.read_ccp4 \n
    Returns: 
    1.-2. UC origin in 1. array relative indices (pixel) 2. (Angstroms)
    3. Voxel dimensions (Angstroms)
    4. abc UC lenths (Angstroms)
    """
    #Lengths in Angstroms for a single voxel(ftp://ftp.wwpdb.org/pub/emdb/doc/Map-format/current/EMDB_map_format.pdf):
    mapxyz = np.zeros((3),dtype=int)
    mapxyz[cdata['MAPC']-1] = 0
    mapxyz[cdata['MAPR']-1] = 1
    mapxyz[cdata['MAPS']-1] = 2
    
    # mapxyz[column_index] = xyz_index
    # originp[xyz_index]   = xyz_pixel_origin
    # originx[xyz_index]   = xyz_origin (angstroms)
    
    # originp = [x, y, z] coordinate of the origin in pixel units
    originp = np.empty((3,), dtype=np.int)
    originp[mapxyz[0]] = cdata['NCSTART']
    originp[mapxyz[1]] = cdata['NRSTART']
    originp[mapxyz[2]] = cdata['NSSTART']
    
    # Voxel dimensions, origin in Angstrom:
    vox     = np.array([ cdata['X']/cdata['NX'], cdata['Y']/cdata['NY'], cdata['Z']/cdata['NZ'] ])
    originx = vox * originp
    abc     = np.array([cdata['X'],cdata['Y'],cdata['Z']])
    return originp, originx, vox, abc

def create_envelope(data, atom_coords, originx, vox, radius, expand=True, return_density=True):
    # radius in pixels
    R         = np.array([radius/vox[0], radius/vox[1], radius/vox[2]])
    if expand is True :
        deltaShape = (np.ceil(R)*2).astype(np.int) + np.array([2,2,2])
        newShape   = np.asarray(data.shape) + deltaShape
        originx    = originx - vox * deltaShape/2
    else :
        newShape = data.shape
    
    mask       = np.zeros( tuple(newShape), dtype=data.dtype )
    
    # atom_coords --> pixel coords in the density map (rounded to int)
    ijk0 = np.rint([ (xyz - originx) / vox for xyz in atom_coords.T]).astype(np.int)
    
    ijk = modulo_operation(ijk0.T, newShape)
    
    # put a 1 at atomic positions
    mask[tuple(ijk)] = 1
    
    # convolve with a gaussian then threshold to get a region selection
    ###################################################################
    # gaussian cutoff that is equiv to R-cutoff
    threshold = 1/(2*np.pi)**(3/2)/R[0]/R[1]/R[2]*np.exp(-3.0/2.0) 
    mask      = gaussian_filter(mask, R) > threshold
    
    if return_density is True :
        if expand is True :
            density = np.zeros(mask.shape, dtype=data.dtype)
            density[deltaShape[0]//2 : -deltaShape[0]//2, \
                    deltaShape[1]//2 : -deltaShape[1]//2, \
                    deltaShape[2]//2 : -deltaShape[2]//2] = data
        else :
            density = data
        #density *= mask
        return mask, originx, density
    else :
        return mask, originx 

def get_map_grid(ccp4, vox = None): 
    # get x, y, z coords for the data along each dimension
    originp, originx, vox0, abc = get_origin_voxel_unit(ccp4)

    if vox is not None :
        data2 = real_space_interpolation(ccp4['data'], vox0, vox)
    else :
        data2 = ccp4['data']
        vox   = vox0
    
    geom = {}
    geom['abc']     = abc
    geom['originx'] = originx
    geom['vox']     = np.array(vox)
    return data2, geom

def real_space_interpolation(data, voxOld, voxNew):
    """
    use grid interpolation to interpolate data on a new grid 
    keeping the edge pixels alligned
    """
    from scipy.interpolate import RegularGridInterpolator
    # make the new grid
    shape = data.shape
    
    # just make three len 3 empty lists
    X, X2, shape2 = [[None for i in range(3)] for j in range(3)]
    for i in range(3):
        # current grid along each axis
        X[i]      = np.arange(shape[i]) * voxOld[i]
        # the new shape with the new grid 
        shape2[i] = int(round(shape[i] * voxOld[i] / voxNew[i]))
        # the new grid along each axis
        X2[i]     = np.linspace(X[i][0], X[i][-1], shape2[i])
    
    shape2 = tuple(shape2) 
    
    # now interpolate onto the desired grid (minor adjustment)
    interp = RegularGridInterpolator(tuple(X), data)
    x,y,z  = np.meshgrid(X2[0], X2[1], X2[2], indexing='ij')
    data2  = interp( np.array([x.ravel(), y.ravel(), z.ravel()]).T ).reshape(shape2)
    return data2

def Fourier_padd_truncate(data, voxOld, voxNew):
    """
    Fourier padd or cut to get the real-space grid approximately at voxNew.

    zero padd data to avoid alliasing: 2N, dx
    dq = 1/(2Nold dxold), Q = 1/dxold
    
    Now truncate Q: Qnew = 1/dxnew = Nnew dq = Nnew / (2Nold dxold) 
                    Nnew = 2Nold dxold / dxnew
                    voxNew2 = 1 / (dq * Nnew) = 2Nold dxold / Nnew

    Normalisation: mean(data) = Data[0] / Nold
    Normalisation: mean(out)  = Out[0]  / 2 Nnew
    """
    # zero padd data to prevent aliasing
    out = np.zeros(tuple(2*np.array(data.shape)), dtype=data.dtype)
    
    slices = [slice(data.shape[i]) for i in range(len(data.shape))]
    out[slices] = data
    
    # Fourier transform
    out  = np.fft.fftn(out)
    Nnew = np.rint(np.array(out.shape) * voxOld / voxNew).astype(np.int)

    for i in range(len(out.shape)):
        out = zero_padd_truncate_fftshifted(out, Nnew[i], i)
    
    out = np.fft.ifftn(out)
    
    slices = [slice(out.shape[i]//2) for i in range(len(out.shape))]
    out = out[slices]
    
    # normalise to keep the same mean value in real space
    out *= out.size / float(data.size)
    
    # calculate the new sampling
    voxNew2 = 2*np.array(data.shape) * voxOld / Nnew
    return out.astype(data.dtype), voxNew2

def zero_padd_truncate_fftshifted(arr, N, axis):
    oldShape = arr.shape
    newShape = np.array(oldShape).copy()
    newShape[axis] = N
    out = np.zeros(newShape, arr.dtype)
    
    # zero padd
    if N > arr.shape[axis]  :
        slices = []
        for i in range(len(out.shape)):
            if i != axis :
                slices.append(slice(None))
            else :
                slices.append(slice(out.shape[i]//2 - arr.shape[i]//2, out.shape[i]//2 + (1+arr.shape[i])//2))
         
        out[slices] = np.fft.fftshift(arr)
    # truncate
    elif N < arr.shape[axis] :
        slices = []
        for i in range(len(out.shape)):
            if i != axis :
                slices.append(slice(None))
            else :
                slices.append(slice(arr.shape[i]//2 - out.shape[i]//2, arr.shape[i]//2 + (1+out.shape[i])//2))
        
        out         = np.fft.fftshift(arr)[slices]
    else :
        out[:] = arr
        
    out = np.fft.ifftshift(out)
    return out



def get_mol_density_ccp4_pdb(ccp4_fnam, pdb_fnam, radius, vox = None):
    # read the ccp4 file: a dictionary 
    ccp4 = ccp4_reader.read_ccp4(ccp4_fnam)
    
    # get the map on the original grid
    data, geom = get_map_grid(ccp4, None)
    
    # read in the xyz coord for the atoms in the 
    # rigid unit
    xyz = pdb_parser.coor(pdb_fnam)
    
    # get the mask of the single rigid unit
    mask, originx, data = create_envelope(data, xyz, geom['originx'], geom['vox'], radius, \
                                          expand=True, return_density=True)
    
    # this is a bit hacky
    #####################
    # Fourier padd / cut to get approximately the correct grid
    data, vox2 = Fourier_padd_truncate(mask * data, geom['vox'], vox)
    geom['vox'] = vox2
    
    # real-space interpolation to get the exact grid
    data = real_space_interpolation(data, vox2, vox)
    geom['vox'] = vox
    
    # now cut again 
    mask, originx, data = create_envelope(data, xyz, geom['originx'], vox, radius, \
                                          expand=True, return_density=True)
        
    geom['originx'] = originx
    return data, mask, geom


def modulo_operation(ijk0,shape):
    '''
    Wraps the array indices back into the array dimensions \n
    shape: tuple (3,) containing the array dimensions \n
    ijk0: int array (3, N) = indices of map-array BEYOND borders \n
    ijk: integer array returned in shape(3, N) = indices of map-array WITHIN borders
    '''
    ijk = [[i % shape[0], j % shape[1], k % shape[2]] for (i, j, k) in ijk0.T]
    ijk = np.rint(np.array(ijk)).astype(np.int)
    return ijk.T

def put_density_in_U(solid_unit, mask, shape, originp):
    # let's get the array relative coordinates for each non-zero voxel
    # ijk value for each non-zero element in cut_density
    ijk_cut = np.where(mask>0.5)
    
    # ijk value relative to the unit-cell origin, for each non-zero element in cut_density
    ijk_rel = tuple([np.rint(ijk_cut[i] + originp[i]).astype(np.int) for i in range(3)])
    
    # Now we have the un-broken rigid-unit in an array
    U       = np.zeros(shape, dtype=solid_unit.dtype)
    
    # ijk value relative to 'big' origin, for each non-zero element in cut_density
    ijk_U   = tuple([ijk_rel[i] % U.shape[i] for i in range(3)])
    
    # put the cut_density in the big array placed correctly w respect to the origin
    U[ijk_U] = solid_unit[ijk_cut]
    return U, ijk_rel

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
        mlab.points3d(atom_coor[0] - originx[0], atom_coor[1] - originx[1], \
                      atom_coor[2] - originx[2], scale_factor=0.5, color=(1.0,1.0,1.0))
    if show:
        mlab.show()
    else:
        mlab.figure()        

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    # check that the output file was specified
    ##########################################
    if args.filename is not None :
        fnam = args.filename
    elif params['output_file'] is not None :
        fnam = params['output_file']
    else :
        raise ValueError('output_file in the ini file is not valid, or the filename was not specified on the command line')
    
    # make the map using phenix (generates a ccp4 file) 
    ccp4_fnam, pdb_fnam = make_map_ccp4(params['pdb_id'])

    # get the solid_unit volume on the desired grid
    density, mask, geom = get_mol_density_ccp4_pdb(ccp4_fnam, pdb_fnam, params['cut_radius_ang'], params['pixel_size_ang'])

    # now put the single rigid unit in the unit-cell
    solid_unit, ijk_rel = put_density_in_U(density*mask, mask, params['shape'], geom['originx']/geom['vox'])

    # make the input
    ################
    unit_cell  = np.rint(geom['abc'] / geom['vox'])
    N          = params['n']
    sigma      = params['sigma']
    del params['n']
    del params['sigma']
    
    # calculate the diffraction data and metadata
    #############################################
    diff, info = forward_sim.generate_diff(solid_unit, unit_cell, N, sigma, **params)

    info['unit_cell_pixels'] = unit_cell
    info['unit_cell_ang']    = geom['abc']
    info['density_cut']      = mask * density
    
    # calculate the constraint ratio
    ################################
    omega_con, omega_Bragg, omega_global  = calculate_constraint_ratio(info['support'], params['space_group'], unit_cell)
    info['omega_continuous'] = omega_con
    info['omega_Bragg']      = omega_Bragg
    info['omega_global']     = omega_global

    # output
    ########
    outputdir = os.path.split(os.path.abspath(args.filename))[0]

    # mkdir if it does not exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    f = h5py.File(fnam)
    
    group = '/forward_model_pdb'
    if group not in f:
        f.create_group(group)
    
    # solid unit
    key = group+'/solid_unit'
    if key in f :
        del f[key]
    f[key] = solid_unit
    
    # diffraction data
    key = group+'/data'
    if key in f :
        del f[key]
    f[key] = diff

    # everything else
    for key, value in info.items():
        if value is None :
            continue 
        
        h5_key = group+'/'+key
        if h5_key in f :
            del f[h5_key]
        
        try :
            print('writing:', h5_key, type(value))
            f[h5_key] = value
        
        except Exception as e :
            print('could not write:', h5_key, ':', e)

    f.close() 
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print(e)
    
    # copy the pdb file to the output directory
    ###########################################
    try :
        import shutil
        pdb_fnam = os.path.abspath(os.path.join('./temp', params['pdbid'] + '.pdb'))
        shutil.copy(pdb_fnam, outputdir)
    except Exception as e :
        print(e)
