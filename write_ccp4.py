import numpy as np
import struct
from array import array

def make_header(map,geom,SGN):
    '''
    Creates ccp4 header as needed for write_ccp4()-function
    Parameters:
    - map : Your data array (dtype=float)
    - geom : dictionary {} containing geometry parameters (as lists or arrays)
        - 'vox' = [x,y,z] voxel dimensions (Angstroms)
        - 'abc' = [a,b,c] UC lenths (Angstroms)
        - 'originx' = [x,y,z] coordinates of map origin relative to UC origin (Angstroms)
        - 'angles' (optional) = [Alpha,Beta,Gamma] angels (deg/): Default is [90.0,90.0,90.]
    - SGN : Space Group number (e.g. P21=4, P212121=19, if you don't know, you may look it up e.g. at 
    https://en.wikipedia.org/wiki/List_of_space_groups)
    '''
    from collections import OrderedDict
    d = OrderedDict()
    #header = struct.unpack('10i6f3i3f3i12f15i4s4Bfi',header)
    abc = geom['abc']
    vox = geom['vox']
    if 'angles' in geom:
        angles = geom['angles']
    else:
        angles = [90.0, 90.0, 90.0]
    originp = np.rint( geom['originx'] / vox ).astype(np.int)
    Nxyz = np.rint(abc/vox).astype(np.int)
    mapc = 3
    mapr = 2
    maps = 1
    print(map.shape,originp)
    d['NC']          = map.shape[mapc-1]
    d['NR']          = map.shape[mapr-1]
    d['NS']          = map.shape[maps-1]
    d['MODE']        = 2
    d['NCSTART']     = originp[mapc-1]
    d['NRSTART']     = originp[mapr-1]
    d['NSSTART']     = originp[maps-1]
    d['NX']          = Nxyz[0]
    d['NY']          = Nxyz[1]
    d['NZ']          = Nxyz[2]
    d['X']           = abc[0]
    d['Y']           = abc[1]
    d['Z']           = abc[2]
    d['Alpha']       = angles[0]
    d['Beta']        = angles[1]
    d['Gamma']       = angles[2]    
    d['MAPC']        = mapc
    d['MAPR']        = mapr
    d['MAPS']        = maps
    d['AMIN']        = np.min(map)
    d['AMAX']        = np.max(map)
    d['AMEAN']       = np.mean(map)
    d['ISPG']        = SGN
    d['NSYMBT']      = 0
    d['LSKFLG']      = 0
    d['SKWMAT']      = np.array([[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.]])
    d['SKWTRN']      = np.array([ 0.,  0.,  0.])
    d['future_use']  = np.zeros((15),dtype=int)
    d['MAP']         = b'MAP '
    d['MACHST']      = ['0x44', '0x41', '0x0', '0x0']
    d['ARMS']        = np.std(map)
    d['NLABL']       = 1
    #header_dict['LABELS']      = header[59]

    return d
def write_ccp4(map, fname, geom, SGN=19):
    '''
    Writes your map data to a file in ccp4 format
    Parameters:
    - map : Your data array (dtype=float)
    - fname : File name
    - geom : dictionary {} containing geometry parameters (as lists or arrays)
        - 'vox' = [x,y,z] voxel dimensions (Angstroms)
        - 'abc' = [a,b,c] UC lenths (Angstroms)
        - 'originx' = [x,y,z] coordinates of map origin relative to UC origin (Angstroms)
        - 'angles' (optional) = [Alpha,Beta,Gamma] angels (deg/): Default is [90.0,90.0,90.]
    - SGN : Space Group number (e.g. P21=4, P212121=19, if you don't know, you may look it up e.g. at 
    https://en.wikipedia.org/wiki/List_of_space_groups)
    '''
    d = make_header(map,geom,SGN)
    with open(fname,'wb') as f: 
        f.write( struct.pack('<10i6f3i3f3i12f15i4s4Bfi',d['NC'],d['NR'],d['NS'],d['MODE'],\
                             d['NCSTART'],d['NRSTART'],d['NSSTART'],d['NX'],d['NY'],d['NZ'],\
                             d['X'],d['Y'],d['Z'],d['Alpha'],d['Beta'],d['Gamma'], \
                             d['MAPC'],d['MAPR'],d['MAPS'],d['AMIN'],d['AMAX'],d['AMEAN'],\
                             d['ISPG'],d['NSYMBT'],d['LSKFLG'], \
                             *d['SKWMAT'].flatten().tolist(),*d['SKWTRN'].tolist(),\
                             *d['future_use'].tolist(),d['MAP'],*[int(i,16) for i in d['MACHST']],d['ARMS'],d['NLABL']) )
        #header_dict['LABELS']      = header[59]
        f.seek(256*4)
        # determine endianness
        if d['MACHST'] == ['0x44', '0x41', '0x0', '0x0']:
            dtype = 'f'
        elif d['MACHST'] == ['0x11', '0x11', '0x0', '0x0']:
            dtype = np.dtype('>f')
        else :
            raise ValueError('cannot determine endianness')
        map = np.transpose(map, (d['MAPS']-1,d['MAPR']-1,d['MAPC']-1))
        data_array = array(dtype,map.flatten())
        data_array.tofile(f)


if __name__ == '__main__':
    import h5py
    f = h5py.File('hdf5/5jdk/5jdk.h5', 'r')
    s0 = f['forward_model/solid_unit'][()].real
    s1 = f['phase/solid_unit'][()].real
    f.close()
    f = h5py.File('hdf5/5jdk/O_merge.h5', 'r')
    smerge = f['solid_unit'][()].real
    f.close()
    # for testing
    dxyz = np.array([0.575625, 0.63875, 1.0959375])
    Uxyz = np.array([36.840, 40.880, 70.140])
    geom = {'vox': dxyz, 'abc': Uxyz, 'originx': np.array([0,0,0])}
    fnam = 'solid_unit_sol.ccp4'
    write_ccp4(s0.astype(np.float32), fnam, geom, SGN=19)
    fnam = 'solid_unit_ret.ccp4'
    write_ccp4(s1.astype(np.float32), fnam, geom, SGN=19)
    fnam = 'solid_unit_merged.ccp4'
    write_ccp4(smerge.astype(np.float32), fnam, geom, SGN=19)
