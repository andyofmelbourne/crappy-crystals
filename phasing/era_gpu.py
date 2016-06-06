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

import era

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
        S = era.choose_N_highest_pixels( (O * O.conj()).real, support)
        S = ap.array(S)
    else :
        support = ap.array(support)
        S = support
    
    
    O    = O.astype(c_dtype)
    O    = ap.array(O)
    O   *= S
    
    I_norm    = ap.sum(mask * I)
    amp       = ap.array(np.sqrt(I.astype(dtype)))
    eMods     = []
    eCons     = []
    
    if background is not None :
        if background is True :
            background = ap.random.random((I.shape)).astype(dtype)
            background[background < 0.1] = 0.1
        else :
            background = ap.sqrt(background)
        rs = None
    
    
    mask = ap.array(mask.astype(np.int))
    
    # unit cell mapper for the no overlap support
    if params['crystal']['space_group'] == 'P1':
        import crappy_crystals.symmetry_operations.P1 as sym_ops 
        sym_ops_obj = sym_ops.P1(params['crystal']['unit_cell'], params['crystal']['unit_cell'], dtype)
    elif params['crystal']['space_group'] == 'P212121':
        import crappy_crystals.symmetry_operations.P212121 as sym_ops 
        sym_ops_obj = sym_ops.P212121(params['crystal']['unit_cell'], params['crystal']['unit_cell'], dtype)

    mapper_unit = sym_ops_obj.solid_syms_real

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
            
            # support projection 
            if type(support) is int :
                S = era.choose_N_highest_pixels( np.array((O * O.conj()).real), support,\
                            params['crystal']['unit_cell'], mapper = mapper_unit)
                S = ap.array(S)
            else :
                S = support
            O = O * S

            if background is not None :
                bt, rs, r_av = era.radial_symetry(np.array(background), rs = rs)
                background   = ap.array(bt)
            
            # metrics
            eCon = l2norm(O, O0)
            
            eMod  = model_error(amp, O, Imap, mask, background = background)
            eMod  = ap.sqrt( eMod / I_norm )
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)

        if full_output : 
            info = {}
            info['I']    = np.array(Imap(ap.fft.fftn(O)))
            info['support'] = np.array(S)
            if background is not None :
                background = np.array(background)
                background, rs, r_av = era.radial_symetry(background**2, rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return np.array(O), info
        else :
            return np.array(O)

def model_error(amp, O, Imap, mask, background = None):
    O   = ap.fft.fftn(O)
    if background is not None :
        M   = ap.sqrt(Imap(O) + background**2)
    else :
        M   = ap.sqrt(Imap(O))
    err = ap.sum( mask * (M - amp)**2 ) 
    return err

def pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
    O = ap.fft.fftn(O)
    O = Pmod(amp, O, Imap(O), mask = mask, alpha = alpha)
    O = ap.fft.ifftn(O)
    return O
    
def Pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
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
