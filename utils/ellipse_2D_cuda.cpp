from pycuda.compiler import SourceModule
gpu_fns = SourceModule("""
#include <pycuda-complex.hpp>
__global__ void map(int n, pycuda::complex<double> *O, pycuda::complex<double> *modes, unsigned int *ii, unsigned int *jj)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        modes[jj[i]] += O[ii[i]];
    }
}

""")
