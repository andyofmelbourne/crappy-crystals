import numpy as np

import pyximport; pyximport.install()
from ellipse_2D_cython import project_2D_Ellipse_cython
#import ellipse_2D_cython
#import project_2D_Ellipse_cython
    
if __name__ == '__main__':
    e0 = 2.0
    e1 = 0.5
    x = 2.0
    y = 1.5
    u, v = project_2D_Ellipse_cython(e0, e1, x, y)
    print(u, v)
