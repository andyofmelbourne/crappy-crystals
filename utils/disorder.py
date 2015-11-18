import numpy as np


def make_exp(sigma, shape):
    # make the B-factor thing
    i, j, k = np.meshgrid(np.fft.fftfreq(shape[0], 1.), \
                          np.fft.fftfreq(shape[1], 1.), \
                          np.fft.fftfreq(shape[2], 1.), indexing='ij')

    exp     = np.exp(-4. * sigma**2 * np.pi**2 * (i**2 + j**2 + k**2))
    return exp
