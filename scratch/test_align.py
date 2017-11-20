import numpy as np
import pyqtgraph as pg
import h5py

def align(O1, O2, order=1):
    con  = np.fft.ifftn( np.fft.fftn(O1) * np.fft.fftn(O2).conj() )
    rmin = np.argmax( (con * con.conj()).real )
    shift = np.array(np.unravel_index(rmin, O2.shape))
    import scipy.ndimage
    out = scipy.ndimage.interpolation.shift(O2.real, shift, mode='wrap', order=1) + 1J*scipy.ndimage.interpolation.shift(O2.imag, shift, mode='wrap', order=1)
    return out

def calc_fid(a,b):
    z = b-a
    return np.sum( (z*z.conj()).real ) 

def test_shift(o, i, j, k):
    O = np.fft.rfftn(o)
    # make the qspace grid
    qi = np.fft.fftfreq(o.shape[0])
    qj = np.fft.fftfreq(o.shape[1])
    qk = np.fft.fftfreq(o.shape[2])[:o.shape[2]/2 + 1]
    
    prI = np.exp(- 2J * np.pi * (i * qi))
    prJ = np.exp(- 2J * np.pi * (j * qj))
    prK = np.exp(- 2J * np.pi * (k * qk))
    phase_ramp = np.multiply.outer(np.multiply.outer(prI, prJ), prK)
    
    return np.fft.irfftn(O * phase_ramp)
    
def post_align(O1, O2):
    o1 = np.fft.rfftn(O1.real)
    o2 = np.fft.rfftn(O2.real)
    
    # make the qspace grid
    qi = np.fft.fftfreq(O1.shape[0])
    qj = np.fft.fftfreq(O1.shape[1])
    qk = np.fft.fftfreq(O1.shape[2])[:O1.shape[2]/2 + 1]
    
    I = np.linspace(-1, 1, 11)
    J = np.linspace(-1, 1, 11)
    K = np.linspace(-1, 1, 11)
    fids = np.zeros( (len(I), len(J), len(K)), dtype=np.float)
    for ii, i in enumerate(I):
        prI = np.exp(- 2J * np.pi * (i * qi))
        for jj, j in enumerate(J):
            prJ = np.exp(- 2J * np.pi * (j * qj))
            for kk, k in enumerate(K):
                prK = np.exp(- 2J * np.pi * (k * qk))
                
                phase_ramp = np.multiply.outer(np.multiply.outer(prI, prJ), prK)
                fids[ii, jj, kk] = calc_fid(o1, o2 * phase_ramp)
    
    l = np.argmin(fids)
    i, j, k = np.unravel_index(l, fids.shape)
    print('lowest error at:', i,j,k, fids[i,j,k])
    i, j, k = I[i], J[j], K[k]
    
    prI = np.exp(- 2J * np.pi * (i * qi))
    prJ = np.exp(- 2J * np.pi * (j * qj))
    prK = np.exp(- 2J * np.pi * (k * qk))
    phase_ramp = np.multiply.outer(np.multiply.outer(prI, prJ), prK)
    
    return np.fft.irfftn(o2 * phase_ramp)


def make_FSC(f1, f2, is_fft_shifted=True, spacing=[1,1,1]):
    shape = f1.shape
    F1 = np.fft.fftn(f1).ravel()
    F2 = np.fft.fftn(f2).ravel()
    
    # get the radial pixel values
    i = np.fft.fftfreq(shape[0], spacing[0])
    j = np.fft.fftfreq(shape[1], spacing[1])
    k = np.fft.fftfreq(shape[2], spacing[2])
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    qs      = np.sqrt(i**2 + j**2 + k**2).ravel()
    rs      = (qs / qs.max() * np.max(shape)).astype(np.int).ravel()
    
    r, i  = np.unique(rs, return_index = True)
    qs = qs[i]
    
    FSC = np.zeros_like(r, dtype=np.complex)
    for i, ri in enumerate(r):
        ind = (rs == ri)
        denom = np.sqrt(np.sum(np.abs(F1[ind])**2) * np.sum(np.abs(F2[ind])**2))
        if denom > 0 :
            FSC[i] = np.sum(F1[ind] * F2[ind].conj()) / denom
    return FSC, r, qs

O0 = h5py.File('../hdf5/pdb/pdb.h5', 'r')['/forward_model_pdb/solid_unit'][()]
O1 = h5py.File('../hdf5/pdb/O_42203.h5', 'r')['solid_unit'][()]

O2 = align(O0, O1)
O3 = post_align(O0.real, O2.real)
O4 = O3.real.copy()
O4[O4<0] = 0
O4 = post_align(O0.real, O4.real)

fid = lambda x : np.sqrt( np.sum( np.abs(x - O0)**2 ) / np.sum(np.abs(O0)**2))

# show the Fourier shell correlation:
spacing = np.array([127.692, 225.403, 306.106]) / 32. * 1.0e-10
plot = pg.plot(make_FSC(O0.real, O1.real, spacing)[0].real)
plot.plot(make_FSC(O0.real, O2.real, spacing)[0].real, pen=pg.mkPen('r'))
plot.plot(make_FSC(O0.real, O3.real, spacing)[0].real, pen=pg.mkPen('g'))
plot.plot(make_FSC(O0.real, O4.real, spacing)[0].real, pen=pg.mkPen('g'))

"""
Os = [] 
fids = []
for i in np.arange(-1, 1, 0.5):
    for j in np.arange(-1, 1, 0.5):
        for k in np.arange(-1, 1, 0.5):
            shift = np.array([i, j, k])
            import scipy.ndimage
            Os.append(scipy.ndimage.interpolation.shift(O.real, shift, mode='wrap', order=1) + 1J*scipy.ndimage.interpolation.shift(O.imag, shift, mode='wrap', order=1))
            Os[-1][Os[-1].real < 0] = 0

            fids.append(fid(Os[-1]))
            print('fidelitly:', round(i,2), round(j, 2), round(k, 2), fids[-1])

            FSC, r, q = make_FSC(O0, Os[-1], spacing)
            FSCs.append(FSC.copy())

            plot.plot(FSC.real)
"""
