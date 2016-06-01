"""
We have one rigid (or solid) unit in the crystal. The unit cell is composed
of symmetry related copies of this rigid unit. Each unit cell of the physical
crystal is the ideal unit cell but with the rigid units randomly translated
from their ideal positions. These displacements must be drawn from a gaussian
probability distribution. 

The measured intensity (after merging into 3 dimensions) is the incoherent
addition of two coherent modes: the diffuse scattering and the brag scattering

o     = solid unit real-space
O     = solid unit Fourier-space
M_i   = i'th symmetry operation
O_i   = M_i . O i'th symmetry related object in the unit cell 
sigma = standard deviation of the rigid unit translations (pixel units)
N     = average number of unit cells in the crystal
gaus  = np.exp(-4. * sigma**2 * np.pi**2 * q**2)
L     = crystal lattice (including the effects of finite crystal size
                         Debye Waller factor ...)
I     = measured merged 3D intensity 
M     = forward predicted intensity
S     = support volume for the rigid unit

o 
 | 
 -- O: fft
    | 
    -- diffuse: incoherent symmetrization * (1 - gaus)
    |    
    | 
    -- brag:    |coherent symmetrization|^2 * N * L * gaus
        |
        -- I = diffuse + brag

Pmod . O = O * sqrt(I / M)

Psup . o = o * S
"""

import numpy as np
import afnumpy as ap
import afnumpy.fft 
import maps_gpu as maps
from   maps import update_progress

import crappy_crystals

def l2norm(array1,array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot  = ap.sum((array1 * array1.conj()).real)
    diff = array1-array2
    return ap.sqrt(ap.sum((diff * diff.conj()).real)/tot)

def ERA(I, iters, support, params, mask = 1, O = None, background = None, method = 1, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', full_output = True):
    if dtype is None :
        dtype   = I.dtype
        c_dtype = (I[0,0,0] + 1J * I[0, 0, 0]).dtype
    
    elif dtype == 'single':
        dtype   = np.float32
        c_dtype = np.complex64

    elif dtype == 'double':
        dtype   = np.float64
        c_dtype = np.complex128

    if O is None :
        O  = (np.random.random((I.shape)) + 0J).astype(c_dtype)
        # support proj
        if type(support) is int :
            S = choose_N_highest_pixels( (O * O.conj()).real, support)
        else :
            S = support
        O = O * S
    
    O    = O.astype(c_dtype)
    O    = ap.array(O)
    
    I_norm    = ap.sum(mask * I)
    amp       = ap.sqrt(I.astype(dtype))
    eMods     = []
    eCons     = []
    
    if background is not None :
        if background is True :
            background = ap.random.random((I.shape)).astype(dtype)
        else :
            background = ap.sqrt(background)
        rs = None
    
    
    mask = ap.array(mask.astype(np.int))
    
    mapper = maps.Mappings(params)
    Imap   = lambda x : mapper.make_diff(solid = x)
    
    # initial error
    #print '\nInitial error: ', l2norm(np.sqrt(mask*I), np.sqrt(mask*Imap(np.fft.fftn(O))))
    
    # method 1
    #---------
    if method == 1 :
        if iters > 0 :
            print '\n\nalgrithm progress iteration convergence modulus error'
        for i in range(iters) :
            O0 = O.copy()
            
            # modulus projection 
            if background is not None :
                O, background  = pmod_back(amp, background, O, Imap, mask, alpha = alpha)
            else :
                O = pmod(amp, O, Imap, mask, alpha = alpha)
            
            O1 = O.copy()
            
            # support projection 
            if type(support) is int :
                S = choose_N_highest_pixels( np.array((O * O.conj()).real), support)
                S = ap.array(S)
            else :
                S = support
            O = O * S

            if background is not None :
                bt, rs, r_av = radial_symetry(np.array(background), rs = rs)
                background   = ap.array(bt)
            
            # metrics
            O2   = O.copy()
            eCon = l2norm(O2, O0)
            
            eMod  = model_error(amp, O, Imap, mask, background = background)
            eMod  = ap.sqrt( eMod / I_norm )
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['I']    = np.array(Imap(ap.fft.fftn(O)))
            info['support'] = S
            if background is not None :
                background, rs, r_av = radial_symetry(np.array(background**2), rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O, info
        else :
            return O

def model_error(amp, O, Imap, mask, background = None):
    O   = ap.fft.fftn(O)
    if background is not None :
        M   = ap.sqrt(Imap(O) + background**2)
    else :
        M   = ap.sqrt(Imap(O))
    err = ap.sum( mask * (M - amp)**2 ) 
    return err

def choose_N_highest_pixels(array, N):
    percent = (1. - float(N) / float(array.size)) * 100.
    thresh  = np.percentile(array, percent)
    support = array > thresh
    # print '\n\nchoose_N_highest_pixels'
    # print 'percentile         :', percent, '%'
    # print 'intensity threshold:', thresh
    # print 'number of pixels in support:', np.sum(support)
    return support
        
def radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = rs.ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, background.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)

    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, rs, r_av

def pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
    O = ap.fft.fftn(O)
    O = Pmod(amp, O, Imap(O), mask = mask, alpha = alpha)
    O = ap.fft.ifftn(O)
    return O
    
def Pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
    print type(O), type(mask), type(amp), type(Imap)
    M    = mask * amp / ap.sqrt(Imap + alpha)
    out  = O * M
    out += (1 - mask) * O
    return out

def pmod_back(amp, background, O, Imap, mask = 1, alpha = 1.0e-10):
    O = ap.fft.fftn(O)
    O, background = Pmod_back(amp, background, O, Imap(O), mask = mask, alpha = alpha)
    O = ap.fft.ifftn(O)
    return O, background
    
def Pmod_back(amp, background, O, Imap, mask = 1, alpha = 1.0e-10):
    M = mask * amp / ap.sqrt(Imap + background**2 + alpha)
    out         = O * M
    background *= M
    out += (1 - mask) * O
    return out, background
