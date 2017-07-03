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
import re

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path = [os.path.join(root, 'utils')] + sys.path

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# hack for import error in pyximport
import time
time.sleep(rank*0.5)
import maps

import io_utils
import duck_3D
import forward_sim
import phasing_3d
import fidelity


def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def phase(mapper, iters_str = '100DM 100ERA', beta=1):
    """
    phase a crappy crystal diffraction volume
    
    Parameters
    ----------
    mapper : object
        A class object that can be used by 3D-phasing, which
        requires the following methods:
            I     = mapper.Imap(modes)   # mapping the modes to the intensity
            modes = mapper.Pmod(modes)   # applying the data projection to the modes
            modes = mapper.Psup(modes)   # applying the support projection to the modes
            O     = mapper.object(modes) # the main object of interest
            dict  = mapper.finish(modes) # add any additional output to the info dict
    
    Keyword Arguments
    -----------------
    iters_str : str, optional, default ('100DM 100ERA')
        supported iteration strings, in general it is '[number][alg][space]'
        [N]DM [N]ERA 1cheshire
    """
    alg_iters = config_iters_to_alg_num(iters_str)
    
    Cheshire_error_map = None
    eMod = []
    eCon = []
    O = mapper.O
    for alg, iters in alg_iters :
        
        print(alg, iters)
        
        if alg == 'ERA':
           O, info = phasing_3d.ERA(iters, mapper = mapper)
         
        if alg == 'DM':
           O, info = phasing_3d.DM(iters, mapper = mapper, beta=beta)
        
        if alg == 'cheshire':
           O, info = mapper.scans_cheshire(O, scan_points=None)
           Cheshire_error_map = info['error_map'].copy()
         
        eMod += info['eMod']
        eCon += info['eCon']

    if Cheshire_error_map is not None :
        info['Cheshire_error_map'] = Cheshire_error_map
    
    O = mapper.object(mapper.modes)
    return O, mapper, eMod, eCon, info

def parse_cmdline_args(default_config='phase.ini'):
    parser = argparse.ArgumentParser(description="phase a crappy crystal from it's diffraction intensity. The results are output into a .h5 file.")
    parser.add_argument('-f', '--filename', type=str, \
                        help="file name of the output *.h5 file to edit / create")
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
    
    # check that the output file was specified
    ################################################
    if args.filename is None and params['output_file'] is not None :
        fnam = params['output_file']
        args.filename = fnam
    
    if args.filename is None :
        raise ValueError('output_file in the ini file is not valid, or the filename was not specified on the command line')
    
    return args, params

def centre_array(O):
    Oh     = np.fft.fftn(O)
    
    phase = np.angle(Oh)
    #phase-= phase[0,0,0] 
    
    import scipy.optimize
    
    i = np.fft.fftfreq(O.shape[0]) 
    j = np.fft.fftfreq(O.shape[1]) 
    k = np.fft.fftfreq(O.shape[2]) 
    i,j,k = np.meshgrid(i,j,k, indexing='ij')
    
    shift = []
    for ii, N in zip((i,j,k), O.shape):
        ii = - 2 * np.pi * ii 
        f      = lambda x : np.sum(np.array(np.gradient(((phase - x*ii) % (2.*np.pi)) - np.pi))**2)
        sol    = scipy.optimize.brute(f, (slice(-N//2, N//2, 2),), \
                                      finish=scipy.optimize.optimize.fmin, full_output=False)
        shift.append(-sol[0])
    
    import scipy.ndimage
    out = scipy.ndimage.interpolation.shift(O.real, shift, mode='wrap', order=1) + 1J*scipy.ndimage.interpolation.shift(O.imag, shift, mode='wrap', order=1)
    return out

def phase_align(O1, O2):
    # check different orientations
    # slices gives us the 8 possibilities
    slices = [(slice(None, None, 2*((i//4)%2)-1), slice(None, None, 2*((i//2)%2)-1), slice(None, None, 2*((i)%2)-1)) for i in range(8)]
    
    errmin = np.inf
    for o in [O2[s].copy() for s in slices]:
        Oc    = centre_array(o)
        delta = O1 - Oc
        err   = np.sum((delta * delta.conj()).real)
        if err < errmin :
            O = Oc.copy()
    return O

def align_Os(O):
    # set the mean phase to zero
    if np.iscomplex(O.ravel()[0]):
        s = np.sum(O)
        phase = np.arctan2(s.imag, s.real)
        O *= np.exp(-1J * phase)
    
    # conjugate
    if np.sum(np.angle(O)) < 0.:
        O = O.conj()
    
    if rank == 0 :
        O = centre_array(O)
    
    # get everyone to align their object with respect to the first
    O0 = comm.bcast(O, root=0)
    
    O = phase_align(O0, O)
    
    return O

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    # make the input
    ################
    if params['input_file'] is None :
        f = h5py.File(args.filename)
    else :
        f = h5py.File(params['input_file'])

    # data
    I = f[params['data']][()]
    
    # solid unit
    if params['solid_unit'] is None :
        solid_unit = None
    else :
        print('loading solid_unit from file...')
        solid_unit = f[params['solid_unit']][()]
    
    # detector mask
    if params['mask'] is None :
        mask = None
    else :
        mask = f[params['mask']][()]
    
    # voxel support
    if params['voxels'] is None :
        voxels = None
    elif type(params['voxels']) != int and params['voxels'][0] == '/'  :
        voxels = f[params['voxels']][()]
    else :
        voxels = params['voxels']
    
    # voxel_sup_blur support
    if params['voxel_sup_blur'] is None :
        voxel_sup_blur = None
    elif type(params['voxel_sup_blur']) != float and params['voxel_sup_blur'][0] == '/'  :
        voxel_sup_blur = f[params['voxel_sup_blur']][()]
    else :
        voxel_sup_blur = params['voxel_sup_blur']
    
    # voxel_sup_blur_frac support
    if params['voxel_sup_blur_frac'] is None :
        voxel_sup_blur_frac = None
    elif type(params['voxel_sup_blur_frac']) != float and params['voxel_sup_blur_frac'][0] == '/'  :
        voxel_sup_blur_frac = f[params['voxel_sup_blur_frac']][()]
    else :
        voxel_sup_blur_frac = params['voxel_sup_blur_frac']
    
    # support update frequency
    if params['support_update_freq'] is None :
        support_update_freq = None
    elif type(params['support_update_freq']) != int and params['support_update_freq'][0] == '/'  :
        support_update_freq = f[params['support_update_freq']][()]
    else :
        support_update_freq = params['support_update_freq']

    # fixed support
    if params['support'] is None or params['support'] is False :
        support = None
    else :
        support = f[params['support']][()]
        
    # Bragg weighting
    if params['bragg_weighting'] is None or params['bragg_weighting'] is False :
        bragg_weighting = None
    else :
        bragg_weighting = f[params['bragg_weighting']][()]

    # Diffuse weighting
    if params['diffuse_weighting'] is None or params['diffuse_weighting'] is False :
        diffuse_weighting = None
    else :
        diffuse_weighting = f[params['diffuse_weighting']][()]

    # Unit cell parameters
    if type(params['unit_cell']) != int and params['unit_cell'][0] == '/'  :
        unit_cell = f[params['unit_cell']][()]
    else :
        unit_cell = params['unit_cell']
    
    
    # make the mapper
    #################
    mapper = maps.Mapper_ellipse(f[params['data']][()], 
                                 Bragg_weighting   = bragg_weighting, 
                                 diffuse_weighting = diffuse_weighting, 
                                 solid_unit        = solid_unit,
                                 voxels            = voxels,
                                 voxel_sup_blur    = voxel_sup_blur,
                                 voxel_sup_blur_frac = voxel_sup_blur_frac,
                                 overlap           = params['overlap'],
                                 support           = support,
                                 support_update_freq = support_update_freq,
                                 unit_cell         = unit_cell,
                                 space_group       = params['space_group'],
                                 alpha             = params['alpha'],
                                 dtype             = params['dtype']
                                 )
    f.close()

    # phase
    #######
    O, mapper, eMod, eCon, info = phase(mapper, params['iters'], params['beta'])

    # merge the results
    ###################
    print('merging Os...',)
    O = align_Os(O)
    O = comm.gather(O) 
    print('Done.')
    
    eMod = np.array(comm.gather(eMod))
    eCon = np.array(comm.gather(eCon))
    
    if rank == 0: 
        O = np.mean([O[i] for i in np.where(eMod[:,-1] > np.mean(eMod[:,-1]))[0]], axis=0)
        
        # calculate the fidelity if we have the ground truth
        ####################################################
        if params['input_file'] is None :
            f = h5py.File(args.filename)
        else :
            f = h5py.File(params['input_file'])
        
        if '/forward_model/solid_unit' in f:
            fids, fids_trans = [], []
            #O = h5py.File('duck_both/duck_both.h5.bak')['/phase/solid_unit'][()]
            for o in mapper.sym_ops.solid_syms_real(O):
                fid, fid_trans = fidelity.calculate_fidelity(f['/forward_model/solid_unit'][()], O)
                fids.append(fid)
                fids_trans.append(fid_trans)
            i         = np.argmin(np.array(fids_trans))
            info['fidelity'] = fids[i]
            info['fidelity_trans'] = fids_trans[i]
        
        # output
        ########
        if params['output_file'] is not None and params['output_file'] is not False :
            filename = params['output_file']
        else :
            filename = args.filename
        
        outputdir = os.path.split(os.path.abspath(filename))[0]

        # mkdir if it does not exist
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        
        print('writing to:', filename)
        f = h5py.File(filename)
        
        group = '/phase'
        if group not in f:
            f.create_group(group)
        
        # solid unit
        key = group+'/solid_unit'
        if key in f :
            del f[key]
        f[key] = O
        
        # real-space crystal
        key = group+'/crystal'
        if key in f :
            del f[key]
        f[key] = mapper.sym_ops.solid_to_crystal_real(O)

        del info['eMod']
        del info['eCon']
        info['eMod'] = eMod
        info['eCon'] = eCon
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
