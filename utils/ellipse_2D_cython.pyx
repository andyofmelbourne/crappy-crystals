from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

def project_2D_Ellipse_cython(double e0, double e1, double x, double y, bint e0_inf = 0, bint e1_inf = 0):
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
    cdef double s0, s1, s, ratio0, ratio1, g, n0, n1, r0, r1

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
        nn = max(n0, n1)
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
