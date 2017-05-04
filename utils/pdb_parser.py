import numpy as np

def uc(fname):
    with open(fname, 'r') as file:
        lines = []
        for line in file:
            lines.append(line)
    for line in lines:
        if line[0:6] == 'CRYST1':
            a=float(line[6:15])
            b=float(line[15:24])
            c=float(line[24:33])
            alpha=float(line[33:40])
            beta=float(line[40:47])
            gamma=float(line[47:54])
    #abc = np.array([a,b,c])
    return a,b,c, np.array([alpha,beta,gamma])

def coor(fname):
    with open(fname, 'r') as file:
        lines = []
        for line in file:
            lines.append(line)
    xo = []
    yo = []
    zo = []
    for line in lines:
        if line[0:6] in ('ATOM  ', 'HETATM'):
            xo.append(float(line[30:37]))
            yo.append(float(line[38:45]))
            zo.append(float(line[46:53]))  
    Xo = np.array([xo,yo,zo])
    return Xo

def fcoor(fname):
    with open(fname, 'r') as file:
        lines = []
        for line in file:
                lines.append(line)
    xo = []
    yo = []
    zo = []
    s = np.ndarray((3,3), dtype=float)
    u = np.ndarray((3), dtype=float)
    for line in lines:
        if line[0:5] == 'ATOM ':
            xo.append(float(line[30:38]))
            yo.append(float(line[38:46]))
            zo.append(float(line[46:54]))  
        if line[0:5] == 'SCALE':    # Transformation matrix (orthogonal - fractional/ crystallographic coordinates)
            n=int(line[5])-1
            s[n][0] = float(line[10:20])
            s[n][1] = float(line[20:30])
            s[n][2] = float(line[30:40])
            u[n] =  float(line[45:55])
    xo = np.array(xo)
    yo = np.array(yo)
    zo = np.array(zo)
    # Transforming orthogonal atom coordinates to fractional crystallographic coordinates
    xfrac = s[0][0]*xo + s[0][1]*yo + s[0][2]*zo + u[0]
    yfrac = s[1][0]*xo + s[1][1]*yo + s[1][2]*zo + u[1]
    zfrac = s[2][0]*xo + s[2][1]*yo + s[2][2]*zo + u[2]
    Xf = np.array([xfrac, yfrac, zfrac])
    
    return Xf

