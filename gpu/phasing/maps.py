import numpy as np
import sys
import pyopencl.clmath

import pyfft
import pyopencl
import pyopencl.array
from   pyfft.cl import Plan
import pyopencl.clmath


class Mappings():
    def __init__(self, duck, support, amp, good_pix, params):
        """ 
        send input numpy arrays to the gpu.
        store needed dummy arrays on the gpu.
        """
        shape = tuple(params['detector']['shape'])
        if params['phasing']['precision'] == 'single':
            dtype = np.float32
            ctype = np.complex64
        elif params['phasing']['precision'] == 'double':
            dtype = np.float64
            ctype = np.complex128
        
        # get the CUDA platform
        platforms = pyopencl.get_platforms()
        for p in platforms:
            if p.name == 'NVIDIA CUDA':
                platform = p
        
        # get one of the gpu's device id
        self.device = platform.get_devices()[0]
        
        # create a context for the device
        self.context = pyopencl.Context([self.device])
        
        # create a command queue for the device
        self.queue = pyopencl.CommandQueue(self.context)
        
        # make a plan for the ffts
        self.plan = Plan(shape, dtype=ctype, queue=self.queue)
        
        # send it to the gpu
        self.duck         = pyopencl.array.to_device(self.queue, np.ascontiguousarray(duck.astype(ctype)))
        self.support      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(support.astype(np.int8)))
        self.amp          = pyopencl.array.to_device(self.queue, np.ascontiguousarray(amp.astype(dtype)))
        self.good_pix     = pyopencl.array.to_device(self.queue, np.ascontiguousarray(good_pix.astype(np.int8)))
        
        # send dummy arrays to the gpu
        self.dummy_real  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(amp.astype(dtype))))
        self.dummy_comp  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(duck.astype(ctype))))

        self.dummy_real2 = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(amp.astype(dtype))))
        self.dummy_comp2 = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(duck.astype(ctype))))

    def _Pmod(self, psi, M, alpha = 1.0e-10):
        self.dummy_real = self.good_pix * self.amp / (pyopencl.clmath.sqrt(M) + alpha)
        self.dummy_comp = psi * self.dummy_real
        
        psi *= (1 - self.good_pix)
        
        psi += self.dummy_comp 
        return psi
    


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
