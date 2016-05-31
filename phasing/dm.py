import numpy as np
import maps
from   maps import update_progress
import era

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
        O  = np.random.random((I.shape)).astype(c_dtype)
        # support proj
        if type(support) is int :
            S = era.choose_N_highest_pixels( (O * O.conj()).real, support)
        else :
            S = support
        O = O * S
    
    O    = O.astype(c_dtype)
    O0   = O.copy()
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    eMods     = []
    eCons     = []
    
    if background is not None :
        if background is True :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            background = np.sqrt(background)
        b0 = background.copy()
        rs = None
    
    
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
                S = era.choose_N_highest_pixels( (O * O.conj()).real, support)
            else :
                S = support
            O0 = O * S

            if background is not None :
                b0, rs, r_av = era.radial_symetry(background, rs = rs)

            O          -= O0
            O0         -= O
            # modulus projection 
            if background is not None :
                background -= b0
                b0         -= background
                O0, b0      = era.pmod_back(amp, b0, O0, Imap, mask, alpha = alpha)
                background += b0
            else :
                O0 = era.pmod(amp, O0, Imap, mask, alpha = alpha)
            O  += O0

            # metrics
            eCon   = l2norm(O_bak, O)
            
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            O0    = O * S
            eMod  = model_error(amp, O0, Imap, mask, background = background)
            eMod  = np.sqrt( eMod / I_norm )

            update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['I']       = Imap(np.fft.fftn(O0))
            info['support'] = S
            if background is not None :
                background, rs, r_av = era.radial_symetry(background**2, rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O0, info
        else :
            return O0

def model_error(amp, O, Imap, mask, background = None):
    O   = np.fft.fftn(O)
    if background is not None :
        M   = np.sqrt(Imap(O) + background**2)
    else :
        M   = np.sqrt(Imap(O))
    err = np.sum( mask * (M - amp)**2 ) 
    return err
