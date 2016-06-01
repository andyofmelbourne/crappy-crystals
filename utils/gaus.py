import numpy as np

def gaus(shape, scale, sig, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs      = np.sqrt(i**2 + j**2 + k**2)
        
    if is_fft_shifted is False :
        rs = np.fft.ifftshift(rs)

    exp = scale * np.exp( - rs / (2. * sig**2))
    return exp
