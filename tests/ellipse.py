import numpy as np

import pyximport; pyximport.install()
from ellipse_2D_cython import project_2D_Ellipse_cython
#import ellipse_2D_cython
#import project_2D_Ellipse_cython
from ellipse_2D_cython_new import project_2D_Ellipse_arrays_cython, project_2D_Ellipse_arrays_cython_test, project_2D_Ellipse_cython

import numexpr as ne
import math
    
def project_DBI(x, y, D, B, I):
    tol2 = 1.0e12

    if D < 0 or B < 0 or I < 0 :
        #print('Warning: one of D, B or I < 0')
        #print('doing nothing about it...')
        return x, y
    
    # divide by zero tolerance
    if D == 0 or (D > 0 and I/D > tol2) :
        e0_inf = True
        e0     = 0.
    else :
        e0_inf = False
        e0     = math.sqrt(I/D)
    
    if B ==0 or (B > 0 and I/B > tol2) :
        e1_inf = True
        e1     = 0.
    else :
        e1_inf = False
        e1     = math.sqrt(I/B)
    
    xp, yp = project_2D_Ellipse_cython(e0, e1, x, y, e0_inf, e1_inf)
    return xp, yp

def score(xp, yp, x, y, D, B, I):
    tol = 1.0e-15

    dist_p  = np.empty_like(xp)
    dist_e  = np.empty_like(xp)
    
    # loop
    it = np.nditer([dist_p, dist_e, xp,yp,x,y,I,B,D], 
                   op_flags = [['writeonly', 'no_broadcast'], ['writeonly', 'no_broadcast'],
                               ['readonly'],['readonly'],['readonly'],
                               ['readonly'],['readonly'],
                               ['readonly'],['readonly']])
    
    for dp, de, xpi, ypi, xi, yi, Ii, Bi, Di in it :
        dp[...] = math.sqrt( (xi-xpi)**2 + (yi-ypi)**2 )
        de[...] = math.fabs(Di*xpi**2 + Bi*ypi**2 - Ii) 
    
    # score:
    print('number of points for which |I - Dx^2 - By^2| > ',tol,':', np.sum(dist_e>tol))
    
def div_safe(a, b):
    """
    compute a / b but return zero for a / 0 
    and a mask of True False values (True == a / b is good 
    and False == a / 0)
    e.g.
    c, mask = div_safe([-1., 0., 1.], [0., 1.0e-350, 2.]) 
    c, mask == [0., 0., 0.5], [False, False, True]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        mask = np.ones(c.shape, dtype=np.bool)
        i = ~ np.isfinite( c )
        c[i] = 0  # -inf inf NaN
        mask[i] = False
    return c, mask


def on_ellipse(xp, yp, e0_inf, e1_inf, I0, I, B, D):
    print('\nall pixels:')
    print('-----------')
    diff = np.abs(D.astype(np.longdouble)*xp.astype(np.longdouble)**2 + \
                  B.astype(np.longdouble)*yp.astype(np.longdouble)**2 - \
                  I.astype(np.longdouble))
    print('mean   |D*xp**2 + B*yp**2 - I| :', np.mean(diff))
    print('median |D*xp**2 + B*yp**2 - I| :', np.median(diff))
    
    i = np.argmax(diff)
    print('\nworst value:','diff', diff[i], 'I',I[i], 'D', D[i], 'B', B[i], 'xp', xp[i], 'yp', yp[i])

    print('\nI > 0 :')
    print('-------')
    print('mean   |D*xp**2 + B*yp**2 - I| :', np.mean(diff[I0 != 1]))
    print('median |D*xp**2 + B*yp**2 - I| :', np.median(diff[I0 != 1]))
    
    print('\ne0_inf and e1_inf == 0 :')
    print('------------------------')
    print('mean   |D*xp**2 + B*yp**2 - I| :', np.mean(diff[(e0_inf == 0)*(e1_inf == 0)]))
    print('median |D*xp**2 + B*yp**2 - I| :', np.median(diff[(e0_inf == 0)*(e1_inf == 0)]))
    return diff


def dist_stats(xp, yp, e0_inf, e1_inf, I0, I, B, D):
    print('\nall pixels:')
    print('-----------')
    dx = xp - x
    dy = yp - y
    dists = np.sqrt(dx.astype(np.longdouble)**2 + dy.astype(np.longdouble)**2)
    print('mean   |rp - r| :', np.mean(dists))
    print('median |rp - r| :', np.median(dists))
    
    print('\nI > 0 :')
    print('-------')
    print('mean   |rp - r| :', np.mean(dists[I>0]))
    print('median |rp - r| :', np.median(dists[I>0]))
    
    print('\ne0_inf and e1_inf == 0 :')
    print('------------------------')
    print('mean   |rp - r| :', np.mean(dists[(e0_inf == 0)*(e1_inf == 0)]))
    print('median |rp - r| :', np.median(dists[(e0_inf == 0)*(e1_inf == 0)]))
    return dists

def make_extreme_ellipses(max_exp = 6):
    extreme_values = [10**i for i in range(-max_exp, max_exp+1, 1)]
    extreme_values.insert(10, 0)
    extreme_values = np.array(extreme_values)
    
    I = extreme_values.copy()
    B = extreme_values.copy()
    D = extreme_values.copy()
    x = extreme_values.copy()
    y = extreme_values.copy()
    
    x, y, I, B, D = np.meshgrid(x, y, I, B, D)
    
    # ravel
    I = I.ravel()
    B = B.ravel()
    D = D.ravel()
    x = x.ravel()
    y = y.ravel()
    
    e0, e0_inf = div_safe(np.sqrt(I), np.sqrt(D))
    e1, e1_inf = div_safe(np.sqrt(I), np.sqrt(B))
    e0_inf = np.array(~e0_inf, dtype=np.uint8)
    e1_inf = np.array(~e1_inf, dtype=np.uint8)
    I0     = np.array(I==0, dtype=np.uint8)
    return x, y, e0, e1, e0_inf, e1_inf, I0, I, B, D

def make_random_ellipses(shape = (20,), scale = 1.0e8):
    extreme_values    = np.random.random((11,))*scale
    extreme_values[0] = 0.
    
    B = extreme_values.copy()
    D = extreme_values.copy()
    x = extreme_values.copy()
    y = extreme_values.copy()
    
    x, y, B, D = np.meshgrid(x, y, B, D)
    
    # make sure that there is a solution
    xp    = np.random.random(x.shape)*np.sqrt(scale)
    xp[0] = 0.
    
    yp    = np.random.random(x.shape)*np.sqrt(scale)
    yp[0] = 0.
    I = D*xp**2 + B*yp**2

    # ravel
    I = I.ravel()
    B = B.ravel()
    D = D.ravel()
    x = x.ravel() - scale / 2.
    y = y.ravel() - scale / 2.
    
    e0, e0_inf = div_safe(np.sqrt(I), np.sqrt(D))
    e1, e1_inf = div_safe(np.sqrt(I), np.sqrt(B))
    e0_inf = np.array(~e0_inf, dtype=np.uint8)
    e1_inf = np.array(~e1_inf, dtype=np.uint8)
    I0     = np.array(I==0, dtype=np.uint8)
    return x, y, e0, e1, e0_inf, e1_inf, I0, I, B, D


if __name__ == '__main__':
    """
    # make sure things still work
    e0 = 2.0
    e1 = 0.5
    x = 2.0
    y = 1.5
    u, v = project_2D_Ellipse_cython(e0, e1, x, y)
    # this should print 1.6 and 0.3
    print(u, v)
    assert u == 1.6
    assert v == 0.3
    """
    import time 
    
    print('\nOLD')
    print('Testing random numbers about 1:')
    print('-------------------------------')
    x, y, e0, e1, e0_inf, e1_inf, I0, I, B, D = make_random_ellipses(shape = (15,), scale=1.0e10)
    mask = np.ones(x.shape, dtype=np.uint8)
    
    t0 = time.time()
    xp, yp = project_2D_Ellipse_arrays_cython(e0, e1, x, y, e0_inf, e1_inf, I0)
    t1 = time.time()
    dt_old = t1-t0
    
    diffs_old = on_ellipse(xp, yp, e0_inf, e1_inf, I0, I, B, D)
    
    dists_old = dist_stats(xp, yp, e0_inf, e1_inf, I0, I, B, D)
    xp_old, yp_old = xp.copy(), yp.copy()

    print('\nNEW')
    t0 = time.time()
    xp, yp = project_2D_Ellipse_arrays_cython_test(x, y, D, B, I, mask)
    t1 = time.time()
    dt_new = t1-t0
    diffs_new = on_ellipse(xp, yp, e0_inf, e1_inf, I0, I, B, D)

    dists_new = dist_stats(xp, yp, e0_inf, e1_inf, I0, I, B, D)

    print('\nComparing distances:')
    print(np.sum(dists_old > dists_new), 'values are better in new') 
    print(np.sum(dists_new > dists_old), 'values are better in old') 
    i = np.argmax(dists_new - dists_old)
    print('\nindex', i, 'is the worst value:')
    print('dist new:', dists_new[i]) 
    print('dist old:', dists_new[i]) 
    print('I', I[i], 'D', D[i], 'B', B[i], 'x', x[i], 'y', y[i])
    print('old xp, yp:', xp_old[i], yp_old[i])
    print('new xp, yp:', xp[i], yp[i])
    
    print('\nComparing diffs')
    print(np.sum(diffs_old > diffs_new), 'values are better in new') 
    print(np.sum(diffs_new > diffs_old), 'values are better in old') 
    i = np.argmax(diffs_new - dists_old)
    print('diff new:', diffs_new[i]) 
    print('diff old:', diffs_new[i]) 
    print('\nindex', i, 'is the worst value:')
    print('I', I[i], 'D', D[i], 'B', B[i], 'x', x[i], 'y', y[i])
    print('old xp, yp:', xp_old[i], yp_old[i])
    print('new xp, yp:', xp[i], yp[i])

    print('\nComparing computation time')
    print('old    :', dt_old)
    print('new    :', dt_new)
    print('old/new:', dt_old / dt_new)

    """
    print('\n\n*******************************')
    print('Testing extreme values')
    print('*******************************')
    print('*******************************')
    for max_exp in range(1, 10):
        print('\nmaximum exponential:', max_exp)
        x, y, e0, e1, e0_inf, e1_inf, I0, I, B, D = make_extreme_ellipses(max_exp)
        
        try :
            xp, yp = project_2D_Ellipse_arrays_cython(e0, e1, x, y, e0_inf, e1_inf, I0)
        except ZeroDivisionError :
            print('Failed for maximum exponential:', max_exp)
            break
        
        on_ellipse(xp, yp, e0_inf, e1_inf, I0, I, B, D)
    """
