import numpy as np
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
        shape = params['detector']['shape']
        dtype = np.dtype(params['phasing']['dtype'])
        
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
        self.plan = Plan(shape, dtype=dtype, queue=self.queue)
        
        # send it to the gpu
        self.duck         = pyopencl.array.to_device(self.queue, np.ascontiguousarray(duck))
        self.support      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(support.astype(np.int8)))
        self.amp          = pyopencl.array.to_device(self.queue, np.ascontiguousarray(amp))
        self.good_pix     = pyopencl.array.to_device(self.queue, np.ascontiguousarray(good_pix.astype(np.int8)))
        
        self.Pmod = self._Pmod
        
        # send dummy arrays to the gpu
        self.dummy_real  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(amp)))
        self.dummy_comp  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(duck)))
        self.dummy_comp2 = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(duck)))
        return self #, psi_gpu, amp_gpu, support_gpu, good_pix_gpu

