import numpy as np
import sys
import os
import ConfigParser
import time
import re
import copy

import crappy_crystals
import crappy_crystals.utils
from crappy_crystals import utils
#from crappy_crystals import phasing
from crappy_crystals.gpu.phasing.maps import Mapper_naive as Mapper_naive_gpu 
from crappy_crystals.phasing.maps import *
#from crappy_crystals.phasing.era import ERA
#from crappy_crystals.phasing.dm  import DM

import phasing_3d

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def out_merge(out, I, good_pix):
    # average the background retrievals
    if out[0]['background'] is not None :
        background = np.mean([i['background'] for i in out], axis=0)
    else :
        background = 0

    silent = True
    if rank == 0: silent = False
    
    # centre, flip and average the retrievals
    O, PRTF    = phasing_3d.utils.merge.merge_sols(np.array([i['O'] for i in out]), silent)
    support, t = phasing_3d.utils.merge.merge_sols(np.array([i['support'] for i in out]).astype(np.float), True)
       
    eMod    = np.array([i['eMod'] for i in out])
    eCon    = np.array([i['eCon'] for i in out])

    # mpi
    if size > 1 :
        O          = comm.gather(O, root=0)
        support    = comm.gather(support, root=0)
        eMod       = comm.gather(eMod, root=0)
        eCon       = comm.gather(eCon, root=0)
        PRTF       = comm.gather(PRTF, root=0)
        if background is not 0 :
            background = comm.gather(background, root=0)
        
        if rank == 0 :
            PRTF           = np.abs(np.mean(np.array(PRTF), axis=0))
            t, t, PRTF_rav = phasing_3d.src.era.radial_symetry(PRTF)
            
            eMod       = np.array(eMod).reshape((size*eMod[0].shape[0], eMod[0].shape[1]))
            eCon       = np.array(eCon).reshape((size*eCon[0].shape[0], eCon[0].shape[1]))
            O, t       = phasing_3d.utils.merge.merge_sols(np.array(O))
            support, t = phasing_3d.utils.merge.merge_sols(np.array(support))
            if background is not 0 :
                background = np.mean(np.array(background), axis=0)
    else :
        PRTF = PRTF_rav = None
        
    if rank == 0 :
        # get the PSD
        PSD, PSD_I, PSD_phase = phasing_3d.utils.merge.PSD(O, I)

        out_m = out[0]
        out_m['I'] = np.abs(np.fft.fftn(O))**2
        out_m['O'] = O
        out_m['background'] = background
        out_m['PSD']      = PSD
        out_m['PSD_I']    = PSD_I
        out_m['PRTF']     = PRTF
        out_m['PRTF_rav'] = PRTF_rav
        out_m['eMod']     = eMod
        out_m['eCon']     = eCon
        out_m['support']  = support
        return out_m
    else :
        return None

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def phase(I, support, params, good_pix = None, sample_known = None):
    d   = {'eMod' : [],         \
           'eCon' : [],         \
           'O'    : None,       \
           'background' : None, \
           'B_rav' : None, \
           'support' : None     \
            }
    out = []

    # move all of the phasing params to the top level
    for k in params.keys():
        if k != 'phasing_parameters':
            params['phasing_parameters'][k] = params[k]

    params['phasing_parameters']['O'] = None
    
    params['phasing_parameters']['mask'] = good_pix
    
    if params['phasing_parameters']['support'] is None :
        params['phasing_parameters']['support'] = support

    
    if params['phasing_parameters']['hardware'] == 'gpu':
        params['phasing_parameters']['Mapper'] = Mapper_naive_gpu
    else :
        params['phasing_parameters']['Mapper'] = Mapper_naive

    params0 = copy.deepcopy(params)
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])

    # Repeats
    #---------------------------------------------
    for j in range(params['phasing']['repeats']):
        out.append(copy.deepcopy(d))
        params = copy.deepcopy(params0)
        
        # for testing
        # params['phasing_parameters']['O'] = sample_known
        for alg, iters in alg_iters :
            
            if alg == 'ERA':
               O, info = phasing_3d.ERA(I, iters, **params['phasing_parameters'])
             
            if alg == 'DM':
               O, info = phasing_3d.DM(I,  iters, **params['phasing_parameters'])
             
            out[j]['O']           = params['phasing_parameters']['O']          = O
            out[j]['support']     = params['phasing_parameters']['support']    = info['support']
            out[j]['eMod']       += info['eMod']
            out[j]['eCon']       += info['eCon']
            
            if 'background' in info.keys():
                out[j]['background']  = params['phasing_parameters']['background'] = info['background'] * good_pix
                out[j]['B_rav']       = info['r_av']
    
        out[j]['I'] = info['I']
        out[j]['eMod'] = np.array(out[j]['eMod'])
        out[j]['eCon'] = np.array(out[j]['eCon'])
    return out

"""
def phase(I, solid_support, params, good_pix = None, solid_known = None):

    # Type of sample support
    if 'support' in params['phasing'].keys() and params['phasing']['support'] == 'voxel_number':
        support = params['voxel_number']['n']
        print 'sample update: voxel number with', support, 'voxels'
    else :
        support = solid_support
        print 'sample update: fixed support with ', np.sum(support), 'voxels'
    
    # background
    background = None
    if 'background' in params['phasing'].keys() :
        if params['phasing']['background'] is True :
            background = True

    print 'background:', background, params['phasing']['background']
    
    d0 = time.time()
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
    
    solid_ret = None
    eMod = []
    for alg, iters in alg_iters :

        print alg, iters

        if alg == 'ERA':
            solid_ret, info = ERA(I, iters, support, params, \
                                  mask = good_pix, O = solid_ret, \
                                  background = background, method = 1, hardware = params['phasing']['hardware'], \
                                  alpha = 1.0e-10, dtype = 'double', full_output = True)
                    
            eMod += info['eMod']
            if 'background' in info.keys():
                background = info['background']
            else :
                info['r_av'] = None
        
        if alg == 'DM':
            solid_ret, info = DM(I, iters, support, params, \
                                  mask = good_pix, O = solid_ret, \
                                  background = background, method = 1, hardware = params['phasing']['hardware'], \
                                  alpha = 1.0e-10, dtype = 'double', full_output = True)

            eMod += info['eMod']
            if 'background' in info.keys():
                background = info['background']
            else :
                info['r_av'] = None

    d1 = time.time()
    print '\n\nTime (s):', d1 - d0
    
    return solid_ret, info['I'], info['support'], info['r_av'], np.array(eMod), np.zeros_like(eMod)
"""


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    kwargs = utils.io_utils.read_input_output_h5(args.input)
    
    print kwargs.keys()
    out = phase(kwargs['data'], kwargs['sample_support'], kwargs['config_file'], \
                        good_pix = kwargs['good_pix'], sample_known = kwargs['solid_unit'])
    
    if len(out) > 1 :
        out = out_merge(out, kwargs['data'], kwargs['good_pix'])
    else :
        out = out[0]

    # write the h5 file 
    fnam = os.path.join(kwargs['config_file']['output']['path'], 'output.h5')
    utils.io_utils.write_input_output_h5(fnam, data = kwargs['data'], \
            data_retrieved = out['I'], sample_support = kwargs['sample_support'], \
            sample_support_retrieved = out['support'], good_pix = kwargs['good_pix'], \
            solid_unit = kwargs['solid_unit'], solid_unit_retrieved = out['O'], modulus_error = out['eMod'], \
            fidelity_error = out['eCon'], config_file = kwargs['config_file_name'], B_rav = out['B_rav'])
