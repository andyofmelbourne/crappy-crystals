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
import maps
from   maps import update_progress

import crappy_crystals
from crappy_crystals.utils.l2norm import l2norm

def ERA(I, iters, support, params, mask = 1, O = None, background = None, method = 1, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', full_output = True):
    if hardware == 'gpu':
        import era_gpu
        return era_gpu.ERA(I, iters, support, params, mask, O, background, method, hardware, alpha, dtype, full_output)
    
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
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    eMods     = []
    eCons     = []
    
    if background is not None :
        if background is True :
            print 'generating random background...'
            background = np.random.random((I.shape)).astype(dtype)
            background[background < 0.1] = 0.1
        else :
            print 'using defined background'
            background = np.sqrt(background)
        rs = None
    else :
        print 'no background'
    
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
                S = choose_N_highest_pixels( (O * O.conj()).real, support, \
                        params['crystal']['unit_cell'], mapper = mapper_unit)
            else :
                S = support
            O = O * S

            if background is not None :
                background, rs, r_av = radial_symetry(background.copy(), rs = rs)
            
            # metrics
            eCon = l2norm(O, O0)
            
            eMod  = model_error(amp, O, Imap, mask, background = background)
            eMod  = np.sqrt( eMod / I_norm )
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['I']    = Imap(np.fft.fftn(O))
            info['support'] = S
            if background is not None :
                background, rs, r_av = radial_symetry(background**2, rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O, info
        else :
            return O

def model_error(amp, O, Imap, mask, background = None):
    O   = np.fft.fftn(O)
    if background is not None :
        M   = np.sqrt(Imap(O) + background**2)
    else :
        M   = np.sqrt(Imap(O))
    err = np.sum( mask * (M - amp)**2 ) 
    return err

def crop_fftwise(a, shape):
    """
    crop a keeping the lowest frequencies:
    e.g.
    crop_fftwise([0,1,2,-3,-2,-1], (3,)) = [0, 1, -1]
    """
    out = None
    for axis, s in enumerate(shape):
        i  = np.fft.fftfreq(a.shape[axis], 1./a.shape[axis]).astype(np.int)
        io = np.fft.fftfreq(s, 1./s).astype(np.int)
        
        index       = [slice(None)]*len(shape)
        if s < a.shape[axis] :
            index[axis] = i[io]
            if out is None :
                out = a[index]
            else :
                out = out[index]
        else :
            index[axis] = io[i]
            
            if out is None :
                shape_out       = list(a.shape)
                shape_out[axis] = s
                out = np.zeros(shape_out, dtype=a.dtype)
                out[index] = a
            else :
                shape_out       = list(out.shape)
                shape_out[axis] = s
                out_t = np.zeros(shape_out, dtype=a.dtype)
                out_t[index] = out
                out = out_t
    return out

def choose_N_highest_pixels(arrayin, N, unit_cell = None, mapper = None):
    """
    If unit_cell is a 3 element list then only include 
    voxels in the unit cell volume. 
    """
    if unit_cell is not None :
        array = crop_fftwise(arrayin, unit_cell)
    else : 
        array = arrayin

    # no overlap constraint
    if mapper is not None :
        syms = mapper(array)
        # if array is not the maximum value
        # of the M symmetry related units 
        # then do not update 
        update_mask = syms[0] == np.max(syms, axis=0) 
    else :
        update_mask = np.ones(array.shape, dtype = np.bool)
    
    percent = (1. - float(N) / float(np.sum(update_mask))) * 100.
    thresh  = np.percentile(array[update_mask], percent)
    support = update_mask * (array > thresh)
    
    if unit_cell is not None :
        support = crop_fftwise(support, arrayin.shape)
    
    print '\n\nchoose_N_highest_pixels'
    print 'percentile         :', percent, '%'
    print 'intensity threshold:', thresh
    print 'number of pixels in support:', np.sum(support)
    return support

def radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int16)
        
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

def radial_average_to_array(r_av, shape, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int16)
    
    if is_fft_shifted is False :
        rs = np.fft.fftshift(rs)
    rs = rs.ravel()

    background = r_av[rs].reshape(shape)
    return background



def pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
    O = np.fft.fftn(O)
    O = Pmod(amp, O, Imap(O), mask = mask, alpha = alpha)
    O = np.fft.ifftn(O)
    return O
    
def Pmod(amp, O, Imap, mask = 1, alpha = 1.0e-10):
    M    = mask * amp / np.sqrt(Imap + alpha)
    out  = O * M
    out += (1 - mask) * O
    return out

def pmod_back(amp, background, O, Imap, mask = 1, alpha = 1.0e-10):
    O = np.fft.fftn(O)
    O, background = Pmod_back(amp, background, O, Imap(O), mask = mask, alpha = alpha)
    O = np.fft.ifftn(O)
    return O, background
    
def Pmod_back(amp, background, O, Imap, mask = 1, alpha = 1.0e-10):
    M = mask * amp / np.sqrt(Imap + background**2 + alpha)
    out         = O * M
    background *= M
    out += (1 - mask) * O
    return out, background
