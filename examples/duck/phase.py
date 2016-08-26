import numpy as np
import ConfigParser
import time
import re
import copy

import phasing_3d

# insert the directory in which this file is being executed from
# into sys.path
import os, sys
sys.path.append(os.path.abspath(__file__)[:-len(__file__)])

import crappy_crystals
import crappy_crystals.utils
from crappy_crystals import utils
from crappy_crystals.phasing.maps import Mapper_naive, Mapper_ellipse


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

    
    # fall-back
    params['phasing_parameters']['Mapper'] = Mapper_naive

    if params['phasing']['mapper'] == 'naive' :
        if params['phasing_parameters']['hardware'] == 'gpu':
            from crappy_crystals.gpu.phasing.maps import Mapper_naive as Mapper_naive_gpu 
            params['phasing_parameters']['Mapper'] = Mapper_naive_gpu
        else :
            params['phasing_parameters']['Mapper'] = Mapper_naive

    elif params['phasing']['mapper'] == 'ellipse' :
        params['phasing_parameters']['Mapper'] = Mapper_ellipse

    params0 = copy.deepcopy(params)
    
    alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
    
    # Repeats
    #---------------------------------------------
    for j in range(params['phasing']['repeats']):
        out.append(copy.deepcopy(d))
        params = copy.deepcopy(params0)
        
        # for testing
        # params['phasing_parameters']['O'] = np.roll(sample_known, -4, 1) #* np.random.random(sample_known.shape)
        for alg, iters in alg_iters :
            
            if alg == 'ERA':
               O, info = phasing_3d.ERA(I, iters, **params['phasing_parameters'])
             
            if alg == 'DM':
               O, info = phasing_3d.DM(I,  iters, **params['phasing_parameters'])
             
            if alg == 'cheshireScan':
               mapper  = params['phasing_parameters']['Mapper'](I, **params['phasing_parameters'])
               O, info = mapper.scans_cheshire(O)
            
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


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    kwargs = utils.io_utils.read_input_output_h5(args.input)
    
    print kwargs.keys()
    out = phase(kwargs['data'], kwargs['sample_support'], kwargs['config_file'], \
                        good_pix = kwargs['good_pix'], sample_known = kwargs['solid_unit'])
    
    out = out[0]

    # write the h5 file 
    fnam = os.path.join(kwargs['config_file']['output']['path'], 'output.h5')
    utils.io_utils.write_input_output_h5(fnam, data = kwargs['data'], \
            data_retrieved = out['I'], sample_support = kwargs['sample_support'], \
            sample_support_retrieved = out['support'], good_pix = kwargs['good_pix'], \
            solid_unit = kwargs['solid_unit'], solid_unit_retrieved = out['O'], modulus_error = out['eMod'], \
            fidelity_error = out['eCon'], config_file = kwargs['config_file_name'], B_rav = out['B_rav'])
