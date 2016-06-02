import numpy as np
from   maps import update_progress
import era
import era_gpu

import afnumpy as ap
import afnumpy.fft 
import maps_gpu as maps

import crappy_crystals
from crappy_crystals.utils.l2norm import l2norm

def DM(I, iters, support, params, mask = 1, O = None, background = None, method = 1, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', full_output = True):
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
    O0   = O.copy()
    
    I_norm    = ap.array(np.sum(mask * I))
    amp       = ap.array(np.sqrt(I).astype(dtype))
    eMods     = []
    eCons     = []
    
    if background is not None :
        if background is True :
            print 'generating random background...'
            background = ap.random.random((I.shape)).astype(dtype)
            background[background < 0.1] = 0.1
        else :
            print 'using defined background'
            background = ap.sqrt(background)
        b0 = background.copy()
        rs = None
    else :
        print 'no background'
    
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
            # reference
            O_bak = O.copy()
            
            # support projection 
            if type(support) is int :
                S = era.choose_N_highest_pixels( np.array((O * O.conj()).real), support)
                S = ap.array(S)
            else :
                S = support
            O0 = O * S

            if background is not None :
                background_t, rs, r_av = era.radial_symetry(np.array(background), rs = rs)
                b0 = ap.array(background_t)

            O          -= O0
            O0         -= O
            # modulus projection 
            if background is not None :
                background -= b0
                b0         -= background
                O0, b0      = era_gpu.pmod_back(amp, b0, O0, Imap, mask, alpha = alpha)
                background += b0
            else :
                O0 = era_gpu.pmod(amp, O0, Imap, mask, alpha = alpha)
            O  += O0

            # metrics
            eCon   = era_gpu.l2norm(O_bak, O)
            
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            O0    = O * S
            eMod  = era_gpu.model_error(amp, O0, Imap, mask, background = background)
            eMod  = ap.sqrt( eMod / I_norm )

            update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['I']       = np.array(Imap(ap.fft.fftn(O0)))
            info['support'] = np.array(S)
            if background is not None :
                background           = np.array(background)
                background, rs, r_av = era.radial_symetry(background**2, rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return np.array(O0), info
        else :
            return np.array(O0)

