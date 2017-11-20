from pycuda.compiler import SourceModule
gpu_fns = SourceModule('''
__global__ void project_2D_Ellipse_arrays_cuda(
               unsigned int ii_max, double *x, double *y, 
               double *Wx, double *Wy, double *I, 
               unsigned char *mask, double *u, double *v)
{
    int flipped;
    int x_inv = 0;
    int y_inv = 0;
    double Ii, Wxi, Wyi, s0, s1, s, ratio0, ratio1, g, n0, n1, r0, r1, xp0, xp2, yp2;
    double e1_sq, e0_sq, one_on_ep1, one_on_ep0, ep0, ep1, tol, z0, z1, Ip, xp, yp, nn;
    
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    tol = 1.0e-10;
    
    for (int ii = index; ii < ii_max; ii += stride){
        Ii  = I[ii];
        Wxi = Wx[ii];
        Wyi = Wy[ii];
        
        if ((mask[ii] == 0) || ((Wxi < tol) && (Wyi < tol))){
            u[ii] = x[ii];
            v[ii] = y[ii];
            continue;
        }
        else if (Ii < tol){ 
            if (Wxi < tol){
                u[ii] = x[ii];
                v[ii] = 0.;
            }
            else if (Wyi < tol){
                u[ii] = 0.;
                v[ii] = y[ii];
            }
            else{
                u[ii] = 0.;
                v[ii] = 0.;
            }
            continue;
        }

        else if (Wxi < tol){
            u[ii] = x[ii];
            if (y[ii] < 0){
                // how do we know if this is safe? 
                v[ii] = -sqrt(Ii)/sqrt(Wyi);
            }
            else{
                v[ii] =  sqrt(Ii)/sqrt(Wyi);
            }
            continue;
        }

        else if (Wyi > tol && Wxi/Wyi < tol){
            yp = sqrt(Ii) / sqrt(Wyi);
            xp = sqrt(Ii) / sqrt(Wxi);
            
            if (x[ii] > xp){
                u[ii] = xp;
            }
            else if (x[ii] < -xp){
                u[ii] = -xp;
            }
            else{
                u[ii] = x[ii];
            }
            if (y[ii] < 0){
                v[ii] = -yp;
            }
            else{
                v[ii] =  yp;
            }
            continue;
        }
        
        else if (Wyi < tol){
            v[ii] = y[ii];
            if (x[ii] < 0){
                u[ii] = -sqrt(Ii)/sqrt(Wxi);
            }
            else{
                u[ii] =  sqrt(Ii)/sqrt(Wxi);
            }
            continue;
        }
        
        else if (Wxi > tol and Wyi/Wxi < tol){
            yp = sqrt(Ii) / sqrt(Wyi);
            xp = sqrt(Ii) / sqrt(Wxi);
            
            if (y[ii] > yp){
                v[ii] = yp;
            }
            else if (y[ii] < -yp){
                v[ii] = -yp;
            }
            else{
                v[ii] = y[ii];
            }
            
            if (x[ii] < 0){
                u[ii] = -xp;
            }
            else{
                u[ii] =  xp;
            }
            continue;
        }

        else if ((abs(x[ii]) < tol) && (Wxi < Wyi)){
            u[ii] = x[ii];
            if (y[ii] < 0){
                v[ii] = -sqrt(Ii)/sqrt(Wyi);
            }
            else{
                v[ii] =  sqrt(Ii)/sqrt(Wyi);
            }
            continue;
        }   
        
        else if ((abs(y[ii]) < tol) and (Wyi < Wxi)){
            v[ii] = y[ii];
            if (x[ii] < 0){
                u[ii] = -sqrt(Ii)/sqrt(Wxi);
            }
            else {
                u[ii] =  sqrt(Ii)/sqrt(Wxi);
            }
            continue;
        }
    
        // transpose axes so that e0 > e1 
        // ------------------------------
        if (Wyi >= Wxi){
            flipped = 0;
            ep0 = sqrt(Ii)/sqrt(Wxi);
            ep1 = sqrt(Ii)/sqrt(Wyi);
            one_on_ep0 = sqrt(Wxi)/sqrt(Ii);
            one_on_ep1 = sqrt(Wyi)/sqrt(Ii);
            e0_sq      = Ii/Wxi;
            e1_sq      = Ii/Wyi;
            r0 = Wyi/Wxi;
            xp = x[ii];
            yp = y[ii];
        }
        else {
            flipped = 1;
            ep0 = sqrt(Ii)/sqrt(Wyi);
            ep1 = sqrt(Ii)/sqrt(Wxi);
            one_on_ep1 = sqrt(Wxi)/sqrt(Ii);
            one_on_ep0 = sqrt(Wyi)/sqrt(Ii);
            e0_sq      = Ii/Wyi;
            e1_sq      = Ii/Wxi;
            r0 = Wxi/Wyi;
            xp = y[ii];
            yp = x[ii];
        }   
        // invert the axes so that all y >= 0
        // ----------------------------------
        if (yp < 0){
            y_inv = 1;
            yp = -yp;
        }
        else{
            y_inv = 0;
        }   
        if (xp < 0){
            x_inv = 1;
            xp = -xp;
        }
        else{
            x_inv = 0;
        }

        xp0 = xp;
            
        if (yp < tol){
            n0 = ep0 * xp;
            n1 = e0_sq - e1_sq;
            if (n0 < n1){
                z0 = n0 / n1;
                xp = ep0 * z0;
                yp = ep1 * sqrt(1. - z0*z0);
                //print('projecting from the x-axis')
            }
            else{
                xp = ep0;
                yp = 0.;
            }
        }
        else{
            z0 = xp * one_on_ep0;
            z1 = yp * one_on_ep1;
            
            g = z0*z0 + z1*z1 - 1.;

            if (abs(g) < tol){
                u[ii] = x[ii];
                v[ii] = y[ii];
                continue;
            }
            
            // in general r = (e's / e_N)**2
            //r0 = (ep0 / ep1)**2
            r1 = 1. ;
            
            //sbar = bisection(r, z, g)
            
            n0 = r0 * z0;
            n1 = r1 * z1;
            s0 = z1 - 1.;
            if (g < 0){ 
                s1 = 0. ;
            }
            else{
                // calculate the 'robust length' of r * z
                nn = max(n0, n1);
                s1 = abs(nn) * sqrt( (n0/nn)*(n0/nn) + (n1/nn)*(n1/nn) ) - 1.;
            }
            s = 0.;
            
            //print('s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1', s0, s1, r0, r1, s+r0, s+r1, ep0, ep1, xp, yp, z1)
            
            for (int i = 0; i < 2074; i ++){
                s = (s0 + s1) / 2.;
                if ((s == s0) || (s == s1)){
                    break;
                }
                ratio0 = n0 / (s+r0);
                ratio1 = n1 / (s+r1);
                g = ratio0*ratio0 + ratio1*ratio1 - 1.;
                if (g > 0.){
                    s0 = s;
                }
                else if (g < 0.){
                    s1 = s;
                }
                else{
                    break;
                }
            }    
            xp = r0 * xp / (s + r0);
            yp = r1 * yp / (s + r1);
        }
        
        if (flipped == 0){
            Ip = Wxi*xp*xp + Wyi*yp*yp;
        }
        else{
            Ip = Wyi*xp*xp + Wxi*yp*yp;
        }
        
        if (abs(Ip - Ii) > tol){
            //print('Ip != 1', Ii, Ip)
            Ip = sqrt(Ii) / sqrt(Ip);
            xp *= Ip;
            yp *= Ip;
        } 
        // compare with Wx=0 projection
        xp2 = xp0;
        if (flipped == 0){
            yp2 = sqrt(I[ii] - Wx[ii]*xp2*xp2)/sqrt(Wy[ii]);
        }
        else{
            yp2 = sqrt(I[ii] - Wy[ii]*xp2*xp2)/sqrt(Wx[ii]);
        }

        // uninvert
        if (y_inv == 1){
            yp  = -yp;
            yp2 = -yp2;
        }
            
        if (x_inv == 1){
            xp  = -xp;
            xp2 = -xp2;
        }
        
        // unflip
        if (flipped){
            g = xp;
            xp = yp;
            yp = g;
            
            g = xp2;
            xp2 = yp2;
            yp2 = g;
        }

        n0 = xp2-x[ii];
        n1 = yp2-y[ii];
        if ((abs(n0) > tol) && (abs(n1) > tol)) {
            nn = max(n0, n1);
            r0 = nn * sqrt((n0/nn)*(n0/nn) + (n1/nn)*(n1/nn));
        }
        else{
            r0 = sqrt(n0*n0 + n1*n1);
        }
        n0 = xp-x[ii];
        n1 = yp-y[ii];
        if ((abs(n0) > tol) && (abs(n1) > tol)){
            nn = max(n0, n1);
            r1 = nn * sqrt((n0/nn)*(n0/nn) + (n1/nn)*(n1/nn));
        }
        else{
            r1 = sqrt(n0*n0 + n1*n1);
        }
        if ((r1 - r0) > tol){
            /* printf("projecting straight down...%g, %g, %g, %g\n", xp, yp, xp2, yp2); */
            xp = xp2;
            yp = yp2;
        }

        u[ii] = xp;
        v[ii] = yp;
    }
}
''')
