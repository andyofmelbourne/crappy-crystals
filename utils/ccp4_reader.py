#!/usr/bin/python
"""
The header is organised as 56 words followed by space for ten 80 character text labels as follows:

 1      NC              # of Columns    (fastest changing in map)
 2      NR              # of Rows
 3      NS              # of Sections   (slowest changing in map)
 4      MODE            Data type
                          0 = envelope stored as signed bytes (from
                              -128 lowest to 127 highest)
                          1 = Image     stored as Integer*2
                          2 = Image     stored as Reals
                          3 = Transform stored as Complex Integer*2
                          4 = Transform stored as Complex Reals
                          5 == 0	
 
                          Note: Mode 2 is the normal mode used in
                                the CCP4 programs. Other modes than 2 and 0
                                may NOT WORK
 
 5      NCSTART         Number of first COLUMN  in map
 6      NRSTART         Number of first ROW     in map
 7      NSSTART         Number of first SECTION in map
 8      NX              Number of intervals along X
 9      NY              Number of intervals along Y
10      NZ              Number of intervals along Z
11      X length        Cell Dimensions (Angstroms)
12      Y length                     "
13      Z length                     "
14      Alpha           Cell Angles     (Degrees)
15      Beta                         "
16      Gamma                        "
17      MAPC            Which axis corresponds to Cols.  (1,2,3 for X,Y,Z)
18      MAPR            Which axis corresponds to Rows   (1,2,3 for X,Y,Z)
19      MAPS            Which axis corresponds to Sects. (1,2,3 for X,Y,Z)
20      AMIN            Minimum density value
21      AMAX            Maximum density value
22      AMEAN           Mean    density value    (Average)
23      ISPG            Space group number
24      NSYMBT          Number of bytes used for storing symmetry operators
25      LSKFLG          Flag for skew transformation, =0 none, =1 if foll
26-34   SKWMAT          Skew matrix S (in order S11, S12, S13, S21 etc) if
                        LSKFLG .ne. 0.
35-37   SKWTRN          Skew translation t if LSKFLG .ne. 0.
                        Skew transformation is from standard orthogonal
                        coordinate frame (as used for atoms) to orthogonal
                        map frame, as
 
                                Xo(map) = S * (Xo(atoms) - t)
 
38      future use       (some of these are used by the MSUBSX routines
 .          "              in MAPBRICK, MAPCONT and FRODO)
 .          "   (all set to zero by default)
 .          "
52          "

53	MAP	        Character string 'MAP ' to identify file tyoe
54	MACHST		Machine stamp indicating the machine tyoe
			which wrote file
55      ARMS            Rms deviation of map from mean density
56      NLABL           Number of labels being used
57-256  LABEL(20,10)    10  80 character text labels (ie. A4 format)
"""
import numpy as np
import struct
import sys

def update_progress(progress, algorithm, i):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
    sys.stdout.flush()


def read_ccp4(mname):
    from collections import OrderedDict
    d = OrderedDict()
    # Reading map data
    with open(mname,'rb') as f:
        # All header in one tuple	
        header = f.read(56*4)
        header = struct.unpack('10i6f3i3f3i12f15i4s4Bfi',header)
        d['NC']          = header[0]
        d['NR']          = header[1]
        d['NS']          = header[2]
        d['MODE']        = header[3]
        d['NCSTART']     = header[4]
        d['NRSTART']     = header[5]
        d['NSSTART']     = header[6]
        d['NX']          = header[7]
        d['NY']          = header[8]
        d['NZ']          = header[9]
        d['X']           = header[10]
        d['Y']           = header[11]
        d['Z']           = header[12]
        d['Alpha']       = header[13]
        d['Beta']        = header[14]
        d['Gamma']       = header[15]
        d['MAPC']        = header[16]
        d['MAPR']        = header[17]
        d['MAPS']        = header[18]
        d['AMIN']        = header[19]
        d['AMAX']        = header[20]
        d['AMEAN']       = header[21]
        d['ISPG']        = header[22]
        d['NSYMBT']      = header[23]
        d['LSKFLG']      = header[24]
        d['SKWMAT']      = np.array(header[25:34]).reshape((3,3))
        d['SKWTRN']      = np.array(header[34:37])
        d['future_use']  = np.array(header[37:52])
        d['MAP']         = np.array(header[52])
        d['MACHST']      = [hex(header[53]), hex(header[54]), hex(header[55]), hex(header[56])]
        d['ARMS']        = header[57]
        d['NLABL']       = header[58]
        #header_dict['LABELS']      = header[59]
        
        f.seek(256*4)
        
        # determine endianness
        if d['MACHST'] == ['0x44', '0x41', '0x0', '0x0']:
            dtype = np.dtype('<f')
        elif d['MACHST'] == ['0x11', '0x11', '0x0', '0x0']:
            dtype = np.dtype('>f')
        else :
            raise ValueError('cannot determine endianness')
        
        # read the raw data 
        emap2 = np.fromfile(f, dtype).reshape((d['NS'], d['NR'], d['NC']))
        # now permute the data to get emap[x, y, z] ordering
        emap2 = np.transpose(emap2, (d['MAPS']-1, d['MAPR']-1, d['MAPC']-1))
        
        # test
        #mapxyz = np.zeros((3),dtype=int)
        #mapxyz[d['MAPC']-1] = 0
        #mapxyz[d['MAPR']-1] = 1
        #mapxyz[d['MAPS']-1] = 2
        #nx = int(header[mapxyz[0]]) 
        #ny = int(header[mapxyz[1]])
        #nz = int(header[mapxyz[2]])
        #emap = np.empty((nx,ny,nz),dtype=float)
        #f.seek(256*4)
        #n = np.zeros((3), dtype=int)
        #i = 0 
        #I = int(header[2])
        #for n[2] in range(int(header[2])):
        #    update_progress(i / float(I), 'reading emap', i)
        #    i    += 1
        #    for n[1] in range(int(header[1])):
        #        for n[0] in range(int(header[0])):
        #            byte = f.read(4)
        #            emap[n[mapxyz[0]]][n[mapxyz[1]]][n[mapxyz[2]]] = struct.unpack('<f',byte)[0] # '<' ... little endian; '>' ... big endian
        # print np.allclose(emap2, emap)
        
        d['data'] = emap2
    return d
