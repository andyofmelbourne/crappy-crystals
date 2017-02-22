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

import phasing_3d

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def phase(mapper, iters_str = '100DM 100ERA'):
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
    
    eMod = []
    eCon = []
    for alg, iters in alg_iters :
        
        print alg, iters
        
        if alg == 'ERA':
           O, info = phasing_3d.ERA(iters, mapper)
         
        if alg == 'DM':
           O, info = phasing_3d.DM(iters, mapper)
         
        eMod += info['eMod']
        eCon += info['eCon']
    
    return O, mapper, eMod, eCon, info
