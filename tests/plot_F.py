import numpy as np
import math

import pyximport; pyximport.install()
from ellipse_2D_cython import project_2D_Ellipse_cython

#B = 2.0**2
#D = 0.5**2
#I = 1.
B = 100.
D = 1.0e-8
I = 1.0e9

e0 = math.sqrt(I/min(B,D))
e1 = math.sqrt(I/max(B,D))

x0 = 1.0e8
y0 = 1000.

tmin = -e1**2 + e1*y0
tmax = -e1**2 + math.sqrt(e0**2*x0**2 + e1**2*y0**2)

t = np.linspace(tmin, tmax, 10000, endpoint=True)
F = (e0*x0/(t+e0**2))**2 + (e1*y0/(t+e1**2))**2 - 1.


tbar = t[np.argmin(np.abs(F))]
xp = e0**2*x0/(tbar+e0**2)
yp = e1**2*y0/(tbar+e1**2)
print(xp, yp)

import pyqtgraph as pg
x = np.linspace(0., e0, 10000, endpoint=False)
y = np.sqrt(I - D*x**2)/math.sqrt(B)
plot = pg.plot(x,y)

plot.plot([0,x0], [0,y0])
plot.plot([x0,xp], [y0,yp], pen=pg.mkPen('r'))

xp2, yp2 = project_2D_Ellipse_cython(e0, e1, x0, y0)
print(xp2, yp2)
plot.plot([x0,xp2], [y0,yp2], pen=pg.mkPen('b'))

