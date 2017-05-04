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

def create_envelope(ccp4, atom_coords, radius, expand=True, return_density=False): 
    """
    Creates a mask using PDB-atomic coordinates convolved with a sphere (radius)
    ccp4 : dictionary containing ccp4-header and map data \n
    atom_coords : (3, N) array containing xyz coordinates of each atom \n
    radius : cutoff radius for mask \n
    expand (Boolean): If true, mask array size will be increased that it fits full envelope \n
    return_density: If set true, will return masked map array in shape of mask
    """
    originp, originx, vox, abc = get_origin_voxel_unit(ccp4)
     
    # radius in pixels
    R         = np.array([radius/vox[0], radius/vox[1], radius/vox[2]])
    if expand == True:
        deltaShape = (np.ceil(R)*2).astype(np.int) + np.array([2,2,2])
        newShape = np.asarray(ccp4['data'].shape) + deltaShape
        mask = np.zeros( tuple(newShape),dtype=ccp4['data'].dtype )
        originp = originp - deltaShape/2
    else:
        mask = np.zeros_like(ccp4['data'])
    
    # atom_coords --> pixel coords in the density map (rounded to int)
    ijk0 = np.rint([xyz / vox - originp for xyz in atom_coords.T]).astype(np.int)
    
    ijk = modulo_operation(ijk0.T,ccp4['data'].shape)
    
    # put a 1 at atomic positions
    mask[tuple(ijk)] = 1
    
    # convolve with a gaussian then threshold to get a region selection
    ###################################################################
    # gaussian cutoff that is equiv to R-cutoff
    threshold = 1/(2*np.pi)**(3/2)/R[0]/R[1]/R[2]*np.exp(-3.0/2.0) 
    mask      = gaussian_filter(mask, R) > threshold
    
    if return_density is True:
        density = np.zeros(mask.shape, dtype=ccp4['data'].dtype)
        density[deltaShape[0]//2 : -deltaShape[0]//2, \
                deltaShape[1]//2 : -deltaShape[1]//2, \
                deltaShape[2]//2 : -deltaShape[2]//2] = ccp4['data']
        density *= mask
        return mask, originp, density
    else :
        return mask, originp

def get_mol_density_ccp4_pdb(ccp4_fnam, pdb_fnam, radius):
    # read the ccp4 file: a dictionary 
    ccp4 = ccp4_reader.read_ccp4(ccp4_fnam)
    # get atomic coords: [[x0,y0,z0], [x1, y1, z1]...]
    xyz = pdb_parser.coor(pdb_fnam)
    mask, new_originp, cut_density = create_envelope(ccp4, xyz, radius, True, return_density=True)
    originp, originx, vox, abc     = get_origin_voxel_unit(ccp4)
    return cut_density, mask, vox, new_originp, abc

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
    
    # get the solid_unit volume
    cut_density, mask, vox, new_originp, abc = get_mol_density_ccp4_pdb(ccp4_fnam, pdb_fnam, params['cut_radius_ang'])

    # let's get the array relative coordinates for each non-zero voxel
    # ijk value for each non-zero element in cut_density
    ijk_cut = np.where(mask>0.5)
    
    # ijk value relative to the unit-cell origin, for each non-zero element in cut_density
    ijk_rel = tuple([np.rint(ijk_cut[i] + new_originp[i]).astype(np.int) for i in range(3)])

    # Now we have the un-broken rigid-unit in an array
    # Let's make an array big enough for 2x2x2 solid_units
    big     = np.zeros(tuple(2*np.array(cut_density.shape)), dtype=cut_density.dtype)
    
    # ijk value relative to 'big' origin, for each non-zero element in cut_density
    ijk_big = tuple([ijk_rel[i] % big.shape[i] for i in range(3)])

    # put the cut_density in the big array placed correctly w respect to the origin
    big[ijk_big] = cut_density[ijk_cut]
    
    # then interpolate in fourier space to get the right grid
    big = np.fft.fftn(big)

    # we want cubic voxels (so stretch short axes)
    pass

    # we want an even multiple of unit cells (so cut / zero padd)
    pass

    solid_unit = big
    
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
    
    f.close() 
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print(e)
