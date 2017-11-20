import numpy as np
import h5py

dx    = 3. / np.array([1.9951875,3.52192187,4.78290625])

f = h5py.File('../hdf5/pdb/pdb.h5', 'r')
s1 = f['forward_model_pdb/solid_unit'][()]


def make_exp(sigma, shape, spacing = [1.,1.,1.]):
    # make the B-factor thing
    i, j, k = np.meshgrid(np.fft.fftfreq(shape[0], spacing[0]), \
                          np.fft.fftfreq(shape[1], spacing[1]), \
                          np.fft.fftfreq(shape[2], spacing[2]), indexing='ij')
    if sigma is np.inf :
        print('sigma is inf setting exp = 0')
        exp     = np.zeros(i.shape, dtype=np.float)
    elif sigma is 0 :
        print('sigma is 0 setting exp = 1')
        exp     = np.ones(i.shape, dtype=np.float)
    else :
        try :
            exp = np.exp(-2. * np.pi**2 * ((sigma[0]*i)**2 +(sigma[1]*j)**2+(sigma[2]*k)**2))
        except TypeError: 
            exp = np.exp(-2. * sigma**2 * np.pi**2 * (i**2 + j**2 + k**2))
    return exp

exp = make_exp(dx, s1.shape)

s = s1.copy()

for i in range(3):
    print(i)
    # blur
    s2 = np.fft.ifftn(np.fft.fftn(np.abs(s)**2)*exp)
    # threshold
    s = s * (s2.real > 0.01*s2.real.max())
