#!/usr/bin/env python

# for python 2 / 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys

from .mappers import Mapper
from .mappers import isValid

try :
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError :
    rank = 0

def ERA(iters, **args):
    """
    Find the phases of 'I' given O using the Error Reduction Algorithm.
    
    Parameters
    ----------
    iters : int
        The number of ERA iterations to perform.
    
    hardware : ('cpu', 'gpu'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu') or a single gpu
        ('gpu'). The numerical results should be identical.
    
    mapper : object, optional, default (phasing_3d.src.mappers.Mapper)
        A mapping class that provides the methods supplied by:
            phasing_3d.src.mappers.Mapper
        If no mapper is supplied then the (above) defualt is used and 'args' are passed
        to this mapper for initialisation.
        Set the Mapper for the single mode (default)
        ---------------------------------------
        this guy is responsible for doing:
          I     = mapper.Imap(modes)   # mapping the modes to the intensity
          modes = mapper.Pmod(modes)   # applying the data projection to the modes
          modes = mapper.Psup(modes)   # applying the support projection to the modes
          O     = mapper.object(modes) # the main object of interest
          dict  = mapper.finish(modes) # add any additional output to the info dict
        ---------------------------------------
    
    Returns
    -------
    O : numpy.ndarray, (U, V, K) 
        The real-space object function after 'iters' iterations of the ERA algorithm.
    
    info : dict
        contains diagnostics:
            
            'I'     : the diffraction pattern corresponding to object above
            'eMod'  : the modulus error for each iteration:
                      eMod_i = sqrt( sum(| O_i - Pmod(O_i) |^2) / I )
            'eCon'  : the convergence error for each iteration:
                      eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        
    Notes 
    -----
    The ERA is the simplest iterative projection algorithm. It proceeds by 
    progressive projections of the exit surface waves onto the set of function that 
    satisfy the:
        modulus constraint : after propagation to the detector the exit surface waves
                             must have the same modulus (square root of the intensity) 
                             as the detected diffraction patterns (the I's).
        
        support constraint : the exit surface waves (W) must be separable into some object 
                                 and probe functions so that W_n = O_n x P.
    
    The 'projection' operation onto one of these constraints makes the smallest change to the set 
    of exit surface waves (in the Euclidean sense) that is required to satisfy said constraint.
    Examples 
    --------
    """
    # set the mapper if it has not been provided
    if isValid('mapper', args) : 
        mapper = args['mapper']

    elif isValid('hardware', args) and args['hardware'] == 'gpu':
        from mappers_gpu import Mapper 
        mapper = Mapper(**args)
    
    else :
        print('using default cpu mapper')
        from mappers import Mapper 
        mapper = Mapper(**args)
    
    eMods     = []
    eCons     = []

    modes  = mapper.modes.copy()

    if iters > 0 and rank == 0 :
        print('\n\nalgrithm progress iteration convergence modulus error')
    
    for i in range(iters) :
        # modulus projection 
        # ------------------
        modes = mapper.Pmod(modes)
        
        modes_mod = modes.copy()
        
        # support projection 
        # ------------------
        modes = mapper.Psup(modes)
        
        # metrics
        #eMod    = mapper.l2norm(modes1, modes0)
        #eMod    = mapper.Emod(modes)
        #eMod    = mapper.eMod
        eMod    = mapper.Emod(modes.copy())
        #eMod = 0
        
        dO   = modes - modes_mod
        eCon = mapper.l2norm(dO, modes)
        
        if rank == 0 : update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
        
        eMods.append(eMod)
        eCons.append(eCon)
    
        mapper.next_iter(modes.copy(), i)
        #try :
        #    mapper.next_iter(modes.copy(), i)
        #except Exception as e :
        #    if i == 0 :
        #        print(e)
    
    info = {}
    info['eMod']  = eMods
    info['eCon']  = eCons
    
    info.update(mapper.finish(mapper.Psup(modes)))
    
    O = info['O']
    return O, mapper, info


def update_progress(progress, algorithm, i, emod, esup):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5} {6} {7}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, emod, esup, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
    sys.stdout.flush()

    
