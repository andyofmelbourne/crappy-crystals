from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import afnumpy
import afnumpy.fft

import numpy as np

import crappy_crystals

def cpu_3Dflip(a):
    b = np.empty_like(a)
    b[:, 0, :]  = a[:, 0, :]
    b[:, 1:, :] = a[:, -1:0:-1, :]
    return b

def cpu_4Dflip(a):
    b = np.empty((2,)+a.shape, a.dtype)
    b[0]           = a
    b[1][:, 0, :]  = a[:, 0, :]
    b[1][:, 1:, :] = a[:, -1:0:-1, :]
    return b

def afnumpy_3Dflip(a):
    b = afnumpy.empty(a.shape, a.dtype)
    b[:, 0, :]  = a[:, 0, :]
    b[:, 1:, :] = a[:, -1:0:-1, :]
    return b

def arrayfire_3Dflip(a):
    b = afnumpy.empty(a.shape, a.dtype)
    b.d_array = afnumpy.arrayfire.data.flip(a.d_array, dim=1)
    b.d_array = afnumpy.arrayfire.data.shift(b.d_array, 0, d1=1)
    return b

def arrayfire_4Dflip(a):
    c1 = afnumpy.arrayfire.data.flip(a.d_array, dim=1)
    c1 = afnumpy.arrayfire.data.shift(c1, 0, d1=1)
    
    b = afnumpy.arrayfire.data.join(3, a.d_array, c1)
    b = afnumpy.arrayfire.data.moddims(b, a.shape[2], a.shape[1], a.shape[0], 2)
    b = afnumpy.asarray(b)
    return b

def cpu_flip(a):
    b = np.empty_like(a)
    b[0]   = a[0]
    b[1 :] = a[-1:0:-1]
    return b

def afnumpy_flip(a):
    b = afnumpy.empty(a.shape, a.dtype)
    b[0]   = a[0]
    b[1 :] = a[-1:0:-1]
    return b

def arrayfire_flip(a):
    b = afnumpy.empty(a.shape, a.dtype)
    b.d_array = afnumpy.arrayfire.data.flip(a.d_array, dim=0)
    b.d_array = afnumpy.arrayfire.data.shift(b.d_array, d0=1)
    return b

def test_basic_flipping():
    # test basic flipping
    print('flipping:')
    shape = (10,)
    a = np.fft.fftfreq(shape[0], 1/shape[0])

    print('cpu      : a      ', a)
    print('cpu      : flip a ', cpu_flip(a))

    b = afnumpy.array(a)
    print('afnumpy  : flip a ', afnumpy_flip(b))
    print('arrayfire: flip a ', arrayfire_flip(b))
    
    print('\n\n')
    print('Now test 3D flipping:')
    shape = (4,4,4)
    a = np.random.random(shape)

    print('cpu      : a      ', a)
    print('cpu      : flip a ', cpu_3Dflip(a))

    b = afnumpy.array(a)
    print('afnumpy  : flip a ', afnumpy_3Dflip(b))
    print('arrayfire: flip a ', arrayfire_3Dflip(b))

    
    print('\n\n')
    print('Time test 3D:')
    shape = (128,128,128)
    a = np.random.random(shape)
    b = afnumpy.array(a)

    import time 
    d0 = time.time()
    for i in range(1000):
        temp1 = cpu_3Dflip(a)
    d1 = time.time()
    
    print('cpu : flip a       ', d1-d0, 's')

    d0 = time.time()
    for i in range(1000):
        temp = afnumpy_3Dflip(b)
    d1 = time.time()
    
    print('afnumpy : flip a   ', d1-d0)

    d0 = time.time()
    for i in range(1000):
        temp = arrayfire_3Dflip(b)
    d1 = time.time()
    
    print('arrayfire : flip a ', d1-d0)

    print('arrayfire == np ?:', np.allclose(temp1, np.array(temp)))

    import time 
    print('\n\n')
    print('Time test 4D:')
    shape = (128,128,128)
    a = np.random.random(shape)
    b = afnumpy.array(a)

    d0 = time.time()
    for i in range(100):
        temp1 = cpu_4Dflip(a)
    d1 = time.time()
    
    print('cpu : flip a       ', d1-d0, 's')

    d0 = time.time()
    for i in range(100):
        temp = arrayfire_4Dflip(b)
    d1 = time.time()
    
    print('arrayfire : flip a ', d1-d0)

    print('arrayfire == np ?:', np.allclose(temp1, np.array(temp)))

def test_P212121():
    # test symmetry operations
    print('\n\n')
    print('Testing solid_syms_Fourier:')
    a = np.arange(4**3).reshape((4,4,4)) + 0J

    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    U = sym_ops.solid_syms_Fourier(a)

    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    U_gpu = sym_ops_gpu.solid_syms_Fourier(afnumpy.array(a))

    obj = 0
    for obj in range(4):
        print('\n\n')
        print('obj:', obj)
        print('solid unit:')
        print(a.real)
        print('U == U_gpu ?', np.allclose(np.array(U_gpu[obj]), U[obj] ))
        print('U')
        print(U[obj].real)
        print('U_gpu')
        print(np.array(U_gpu[obj]).real)

    print('\n\n')
    print('all')
    print('U == U_gpu ?', np.allclose(np.array(U_gpu), U ))

    print('\n\n')
    print('Testing solid_syms_real')
    U     = sym_ops.solid_syms_real(a)
    U_gpu = sym_ops_gpu.solid_syms_real(afnumpy.array(a))
    print('U == U_gpu ?', np.allclose(np.array(U_gpu), U ))


    import time 

    shape = (128,128,128)
    a = np.random.random(shape) + 1J * np.random.random(shape)
    b = afnumpy.array(a)

    print('\n\n')
    print('Time test P212121:', shape)

    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)

    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)

    d0 = time.time()
    for i in range(100):
        temp1 = sym_ops.solid_syms_Fourier(a)
    d1 = time.time()
    
    print('cpu : flip a        ', d1-d0, 's')

    d0 = time.time()
    for i in range(100):
        temp = sym_ops_gpu.solid_syms_Fourier(b)
    d1 = time.time()
    
    print('arrayfire : sym_ops ', d1-d0)

    print('arrayfire == np ?:', np.allclose(temp1, np.array(temp)))


def test_P1():
    # test symmetry operations
    print('\n\n')
    print('Testing solid_syms_Fourier:')
    a = np.arange(4**3).reshape((4,4,4)) + 0J

    sym_ops = crappy_crystals.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)
    U = sym_ops.solid_syms_Fourier(a)

    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)
    U_gpu = sym_ops_gpu.solid_syms_Fourier(afnumpy.array(a))

    obj = 0
    print('\n\n')
    print('solid unit:')
    print(a.real)
    print('U == U_gpu ?', np.allclose(np.array(U_gpu), U ))
    print('U')
    print(U.real)
    print('U_gpu')
    print(np.array(U_gpu).real)

    print('\n\n')
    print('Testing solid_syms_real')
    U     = sym_ops.solid_syms_real(a)
    U_gpu = sym_ops_gpu.solid_syms_real(afnumpy.array(a))
    print('U == U_gpu ?', np.allclose(np.array(U_gpu), U ))

    import time 

    shape = (128,128,128)
    a = np.random.random(shape) + 1J * np.random.random(shape)
    b = afnumpy.array(a)

    print('\n\n')
    print('Time test P1:', shape)

    sym_ops = crappy_crystals.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)

    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)

    d0 = time.time()
    for i in range(100):
        temp1 = sym_ops.solid_syms_Fourier(a)
    d1 = time.time()
    
    print('cpu : flip a        ', d1-d0, 's')

    d0 = time.time()
    for i in range(100):
        temp = sym_ops_gpu.solid_syms_Fourier(b)
    d1 = time.time()
    
    print('arrayfire : sym_ops ', d1-d0)

    print('arrayfire == np ?:', np.allclose(temp1, np.array(temp)))

def test_P1_unflips():
    # test mode inversion
    print('\n\n')
    print('Testing --> modes --> object:')
    a = np.arange(4**3).reshape((4,4,4)) + 0J

    sym_ops = crappy_crystals.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)
    U = sym_ops.solid_syms_Fourier(a)

    U_inv = sym_ops.unflip_modes_Fourier(U)

    print('solid == unflipped modes? (for all modes):')
    [print(np.allclose(a, m)) for m in U_inv]

    print('solid == average unflipped modes?:')
    print(np.allclose(a, np.mean(U_inv, axis=0)))

    print('\n')
    print('gpu:')
    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P1(a.shape, a.shape, a.dtype)
    U_gpu = sym_ops_gpu.solid_syms_Fourier(afnumpy.array(a))

    U_inv = sym_ops.unflip_modes_Fourier(U_gpu)

    print('solid == unflipped modes? (for all modes):')
    [print(np.allclose(a, np.array(m))) for m in U_inv]

    print('solid == average unflipped modes?:')
    print(np.allclose(a, np.array(afnumpy.mean(U_inv, axis=0))))

def test_P212121_unflips():
    # test mode inversion
    print('\n\n')
    print('Testing --> modes --> object:')
    a = np.arange(4**3).reshape((4,4,4)) + 0J

    #sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121( \
            (a.shape[0]//2, a.shape[1]//2, a.shape[2]//2), a.shape, a.dtype)
    U = sym_ops.solid_syms_Fourier(a)

    U_inv = sym_ops.unflip_modes_Fourier(U)

    print('solid == unflipped modes? (for all modes):')
    [print(np.allclose(a, m)) for m in U_inv]

    print('solid == average unflipped modes?:')
    print(np.allclose(a, np.mean(U_inv, axis=0)))

    print('modes == broadcast average unflipped modes?:')
    print(np.allclose(U, sym_ops.solid_syms_Fourier(np.mean(U_inv, axis=0))))

    print('\n')
    print('gpu:')
    sym_ops_gpu = crappy_crystals.gpu.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    U_gpu = sym_ops_gpu.solid_syms_Fourier(afnumpy.array(a))

    U_inv = sym_ops.unflip_modes_Fourier(U_gpu)

    print('solid == unflipped modes? (for all modes):')
    [print(np.allclose(a, np.array(m))) for m in U_inv]

    print('solid == average unflipped modes?:')
    print(np.allclose(a, np.array(afnumpy.mean(U_inv, axis=0))))


def test_P212121_Psup():
    # test mode inversion
    print('\n\n')
    print('Testing --> modes --> object:')
    shape = (32,32,32)
    a = np.random.random(shape) + 1J * np.random.random(shape)

    #sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121( \
            (a.shape[0]//2, a.shape[1]//2, a.shape[2]//2), a.shape, a.dtype)

    modes = np.zeros( (2 * sym_ops.syms.shape[0],) + sym_ops.syms.shape[1:], a.dtype)
    # diffuse terms
    modes[:modes.shape[0]//2] = sym_ops.solid_syms_Fourier(a, apply_translation = False)
    # unit cell terms
    modes[modes.shape[0]//2:] = sym_ops.solid_syms_Fourier(a, apply_translation = True)

    out = np.empty_like(modes)

    # diffuse terms: unflip the modes
    out[: out.shape[0]//2] = \
            sym_ops.unflip_modes_Fourier(modes[: modes.shape[0]//2], apply_translation = False)

    # unit_cell terms: unflip the modes
    out[modes.shape[0]//2 :] = \
            sym_ops.unflip_modes_Fourier(modes[modes.shape[0]//2 :], apply_translation = True)

    # average 
    out = np.mean(out, axis=0)

    print('solid == average unflipped modes?:')
    print(np.allclose(a, out))

    print('modes == broadcast average unflipped modes?:')
    modes_out = np.empty_like(modes)
    modes_out[: modes_out.shape[0]//2] = sym_ops.solid_syms_Fourier(out, apply_translation=False)
    modes_out[modes_out.shape[0]//2 :] = sym_ops.solid_syms_Fourier(out, apply_translation=True)

    print(np.allclose(modes, modes_out))

def test_real_Fourier_syms():
    # test mode inversion
    print('\n\n')
    print('Testing Fourier solid --> modes == Real solid --> modes ')
    shape = (32,32,32)
    aR = np.random.random(shape) + 1J * np.random.random(shape)

    a = np.fft.fftn(aR)
    
    #sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121( \
            (a.shape[0]//2, a.shape[1]//2, a.shape[2]//2), a.shape, a.dtype)

    # unit cell terms
    symsF = sym_ops.solid_syms_Fourier(a, apply_translation = True)

    a = aR
    
    #sym_ops = crappy_crystals.phasing.symmetry_operations.P212121(a.shape, a.shape, a.dtype)
    sym_ops = crappy_crystals.phasing.symmetry_operations.P212121( \
            (a.shape[0]//2, a.shape[1]//2, a.shape[2]//2), a.shape, a.dtype)

    # unit cell terms
    symsR = sym_ops.solid_syms_real(a)

    symsR = np.fft.fftn(symsR, axes=(1,2,3))

    print('F{symsR} == symsF ?:', np.allclose(symsR, symsF))



if __name__ == '__main__':
    """
    """
    #test_P212121_unflips()
    #test_P212121_Psup()
    test_real_Fourier_syms()

