from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

cdef inline float float_max(float a, float b): return a if a >= b else b
#cdef inline float float_min(float a, float b): return a if a <= b else b
#float_max = max


def project_2D_Ellipse_cython(double e0, double e1, double x, double y, bint e0_inf = 0, bint e1_inf = 0, bint I0 = 0):
    """
    Solve the ellipse projection problem for a 2D ellipse.
    
    Find the closest point to (x, y) that falls on the elliptical
    surface:
        (x/e0)**2 + (y/e1)**2 = 1
        
    In other words find the vector (u, v) that minimises:
        e = (u-x)**2 + (v-y)**2
    such that:
        (x/e0)**2 + (y/e1)**2 = 1
    
    using a bisection search algorithm. 
    
    Parameters
    ----------
    e0 : scalar
         The length of the ellipse axis at y=0
    
    e1 : scalar
         The length of the ellipse axis at x=0
         
    x : scalar
         The x-coordinate of the intitial point
         
    y : scalar
         The y-coordinate of the intitial point
         
    e0_inf : int
         If the inverse of e0 is very small so that
         the equation of our ellipse becomes:
             (y/e1)**2 = 1
         Then set v = +- e1 (+- depending on initial value 
         of y) and u = x.
         
    e1_inf : int
         If the inverse of e1 is very small so that
         the equation of our ellipse becomes:
             (x/e0)**2 = 1
         Then set u = +- e0 (+- depending on initial value 
         of y) and v = y. If both e0_inf and e1_inf are zero 
         then return u = x and v = y.
    
    Returns
    -------
    u : scalar
        The x-coordinate of the closest point to (x, y) that
        lies on the ellipse surface.
    
    v : scalar
        The y-coordinate of the closest point to (x, y) that
        lies on the ellipse surface.
    """
    cdef int i
    cdef double s0, s1, s, ratio0, ratio1, g, n0, n1, r0, r1, u, v

    #if I0 :
    #    return 0., 0.
    
    if e0_inf and e1_inf :
        return x, y
    
    elif e0_inf :
        u = x
        if y < 0 :
            v = -e1
        else :
            v = e1
        return u, v
            
    elif e1_inf :
        v = y
        if x < 0 :
            u = -e0
        else :
            u = e0
        return u, v
    
    elif e0 == 0.0 and e1 == 0.0 :
        u = 0.0
        v = 0.0
        return u, v
    
    elif e0 == 0.0 :
        u = 0.0
        v = y
        return u, v
    
    elif e1 == 0.0 :
        u = x
        v = 0.0
        return u, v
        
        
    # sort axes so that e0 > e1 
    # -------------------------
    if e0 > e1 :
        flipped = False
        ep0 = e0
        ep1 = e1
        xp = x
        yp = y
    else :
        flipped = True
        ep0 = e1
        ep1 = e0
        xp = y
        yp = x
        
    # invert the axes so that all y >= 0
    # --------------------------------
    x_inv = False ; y_inv = False 
    if yp < 0 :
        y_inv = True
        yp = -yp
        
    if xp < 0 :
        x_inv = True
        xp = -xp        
        
    # change variables
    # ----------------
    # if x or y < tol then clip
    tol = 1.0e-10
    if yp < tol : yp = tol
    if xp < tol : xp = tol
 
    z0 = xp / ep0
    z1 = yp / ep1
    
    g = z0*z0 + z1*z1 - 1.
    
    # in general r = (e's / e_N)**2
    r0 = (ep0 / ep1)**2
    r1 = 1. 
    
    #sbar = bisection(r, z, g)
    
    # Bisection
    # ---------
    """
    Computes s, where g[s] = 0
    g[s] = (r0 z0 / (s + r0))^2 + (r1 z1 / (s + r1))^2  + ... -1
    
    where s is in the interval
    -1 + zN < s < -1 + sqrt(r^2 z0^2 + r1^2 z1^2 + ... )
    
    using the bisection method
    """
    n0 = r0 * z0
    n1 = r1 * z1
    s0 = z1 - 1.
    if g < 0 : 
        s1 = 0. 
    else  :
        # calculate the 'robust length' of r * z
        nn = float_max(n0, n1)
        s1 = abs(nn) * sqrt( (n0/nn)**2 + (n1/nn)**2 ) -1.
    s = 0.
    
    #print('s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1', s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1)

    for i in range(2074): # 1074, 149 for double, single precision
        s = (s0 + s1) / 2.
        if s == s0 or s == s1 :
            break
        ratio0 = n0 / (s+r0)
        ratio1 = n1 / (s+r1)
        g = ratio0**2 + ratio1**2 - 1.
        if g > 0. :
            s0 = s
        elif g < 0.:
            s1 = s
        else :
            break
    
    u = r0 * xp / (s + r0)
    v = r1 * yp / (s + r1)
    
    #print(x_out, y_out, xp, yp, ep0, ep1, g)
    
    # do an additional projection onto the ellipse surface
    # for numerical stability when xp or yp ~ 0
    I = sqrt((u/ep0)**2 + (v/ep1)**2)
    u /= I
    v /= I
    
    # uninvert
    if y_inv :
        v = -v
        
    if x_inv :
        u = -u  
    
    # unflip
    if flipped :
        return v, u
    else :
        return u, v


import numpy as np
cimport numpy as np

ctypedef np.int_t Ctype_int
ctypedef np.float_t Ctype_float
ctypedef np.uint8_t Ctype_bool

cimport cython
#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
def project_2D_Ellipse_arrays_cython(np.ndarray[Ctype_float, ndim=1] e0, 
                                     np.ndarray[Ctype_float, ndim=1] e1,
                                     np.ndarray[Ctype_float, ndim=1] x,
                                     np.ndarray[Ctype_float, ndim=1] y,
                                     np.ndarray[Ctype_bool, ndim=1] e0_inf,
                                     np.ndarray[Ctype_bool, ndim=1] e1_inf,
                                     np.ndarray[Ctype_bool, ndim=1] I0):
    cdef int i, flipped
    cdef unsigned int ii
    cdef int x_inv = 0
    cdef int y_inv = 0
    cdef unsigned int ii_max = <unsigned int> e0.shape[0]
    cdef double s0, s1, s, ratio0, ratio1, g, n0, n1, r0, r1
    cdef double ep0, ep1, tol, z0, z1, I, xp, yp, nn
    cdef np.ndarray[Ctype_float, ndim = 1] u = np.empty((ii_max), dtype=np.float)
    cdef np.ndarray[Ctype_float, ndim = 1] v = np.empty((ii_max), dtype=np.float)
    
    for ii in range(ii_max):
        #if I0[ii] == 1 :
        #    u[ii] = 0.
        #    v[ii] = 0.
        #    continue
        
        if e0_inf[ii] == 1 and e1_inf[ii] == 1 :
            u[ii] = x[ii]
            v[ii] = y[ii]
            continue
        
        elif e0_inf[ii] == 1 :
            u[ii] = x[ii]
            if y[ii] < 0 :
                v[ii] = -e1[ii]
            else :
                v[ii] = e1[ii]
            continue
                
        elif e1_inf[ii] == 1 :
            v[ii] = y[ii]
            if x[ii] < 0 :
                u[ii] = -e0[ii]
            else :
                u[ii] = e0[ii]
            continue
        
        elif e0[ii] == 0.0 and e1[ii] == 0.0 :
            u[ii] = 0.0
            v[ii] = 0.0
            continue
        
        elif e0[ii] == 0.0 :
            u[ii] = 0.0
            v[ii] = y[ii]
            continue
        
        elif e1[ii] == 0.0 :
            u[ii] = x[ii]
            v[ii] = 0.0
            continue
            
            
        # sort axes so that e0 > e1 
        # -------------------------
        if e0[ii] > e1[ii] :
            flipped = 0
            ep0 = e0[ii]
            ep1 = e1[ii]
            xp = x[ii]
            yp = y[ii]
        else :
            flipped = 1
            ep0 = e1[ii]
            ep1 = e0[ii]
            xp = y[ii]
            yp = x[ii]
            
        # invert the axes so that all y >= 0
        # --------------------------------
        #x_inv = False ; y_inv = False 
        if yp < 0 :
            y_inv = 1
            yp = -yp
            
        if xp < 0 :
            x_inv = 1
            xp = -xp        
            
        # change variables
        # ----------------
        # if x or y < tol then clip
        tol = 1.0e-10
        if yp < tol : yp = tol
        if xp < tol : xp = tol
     
        z0 = xp / ep0
        z1 = yp / ep1
        
        g = z0*z0 + z1*z1 - 1.
        
        # in general r = (e's / e_N)**2
        r0 = (ep0 / ep1)**2
        r1 = 1. 
        
        #sbar = bisection(r, z, g)
        
        n0 = r0 * z0
        n1 = r1 * z1
        s0 = z1 - 1.
        if g < 0 : 
            s1 = 0. 
        else  :
            # calculate the 'robust length' of r * z
            nn = float_max(n0, n1)
            s1 = abs(nn) * sqrt( (n0/nn)**2 + (n1/nn)**2 ) -1.
        s = 0.
        
        #print('s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1', s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1)
        
        for i in range(2074): # 1074, 149 for double, single precision
            s = (s0 + s1) / 2.
            if s == s0 or s == s1 :
                break
            ratio0 = n0 / (s+r0)
            ratio1 = n1 / (s+r1)
            g = ratio0**2 + ratio1**2 - 1.
            if g > 0. :
                s0 = s
            elif g < 0.:
                s1 = s
            else :
                break
        
        xp = r0 * xp / (s + r0)
        yp = r1 * yp / (s + r1)
        
        #print(x_out, y_out, xp, yp, ep0, ep1, g)
        
        # do an additional projection onto the ellipse surface
        # for numerical stability when xp or yp ~ 0
        I = sqrt((xp/ep0)**2 + (yp/ep1)**2)
        if I != 1. :
            xp /= I
            yp /= I
        
        # uninvert
        if y_inv :
            yp = -yp
            
        if x_inv :
            xp = -xp
        
        # unflip
        if flipped :
            g = xp
            xp = yp
            yp = g
        
        u[ii] = xp
        v[ii] = yp
    return u, v
