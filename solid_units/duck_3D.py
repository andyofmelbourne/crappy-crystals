import numpy as np

def interp_3d(array, shapeout):
    from scipy.interpolate import griddata
    ijk = np.indices(array.shape)
    
    points = np.zeros((array.size, 3), dtype=np.float)
    points[:, 0] = ijk[0].ravel()
    points[:, 1] = ijk[1].ravel()
    points[:, 2] = ijk[2].ravel()
    values = array.astype(np.float).ravel()

    gridout  = np.mgrid[0: array.shape[0]-1: shapeout[0]*1j, \
                        0: array.shape[1]-1: shapeout[1]*1j, \
                        0: array.shape[2]-1: shapeout[2]*1j]
    arrayout = griddata(points, values, (gridout[0], gridout[1], gridout[2]), method='nearest')
    return arrayout
    

def make_3D_duck(shape = (12, 25, 30)):
    # call in a low res 2d duck image
    duck = np.fromfile('solid_units/duck_300_211_8bit.raw', dtype=np.int8).reshape((211, 300))
    
    # convert to bool
    duck = duck < 50

    # make a 3d volume
    duck3d = np.zeros( (100,) + duck.shape , dtype=np.bool)

    # loop over the third dimension with an expanding circle
    i, j = np.mgrid[0 :duck.shape[0], 0 :duck.shape[1]]

    origin = [150, 150]

    r = np.sqrt( ((i-origin[0])**2 + (j-origin[1])**2).astype(np.float) )

    rs = range(50) + range(50, 0, -1)
    rs = np.array(rs) * 200 / 50.
    
    circle = lambda ri : r < ri
    
    for z in range(duck3d.shape[0]):
        duck3d[z, :, :] = circle(rs[z]) * duck

    # now interpolate the duck onto the required grid
    duck3d = interp_3d(duck3d, shape)
    return duck3d
        
if __name__ == '__main__':
    duck3d = make_3D_duck()
