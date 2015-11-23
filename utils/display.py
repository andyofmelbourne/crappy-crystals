import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import signal
import sys

def show_vol(map_3d):
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)
    ex  = Show_vol(map_3d)
    sys.exit(app.exec_())

class Show_vol(QtGui.QWidget):
    def __init__(self, map_3d):

        super(Show_vol, self).__init__()
        # 3D plot for psi
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.show()

        # layout
        vlayout = QtGui.QVBoxLayout() 
        vlayout.addWidget(self.w)

        data = map_3d
        d = np.empty(data.shape + (4,), dtype=np.ubyte)

        # white scale
        dis   = 255. #(data.astype(np.float) * (255./data.max())).astype(np.ubyte)
        alpha = (data.astype(np.float) * (255./data.max())).astype(np.ubyte)

        d[..., 0] = dis
        d[..., 1] = dis
        d[..., 2] = dis
        d[..., 3] = alpha #((data/data.max())**2 * 255.).astype(np.ubyte)

        # show the x-y-z axis
        d[:, 0, 0] = [255,0,0,100]
        d[0, :, 0] = [0,255,0,100]
        d[0, 0, :] = [0,0,255,100] 
        self.v = gl.GLVolumeItem(d)
        self.v.translate(-data.shape[0]/2,-data.shape[1]/2,-data.shape[2]/2)
        self.w.addItem(self.v)
        ax = gl.GLAxisItem()
        self.w.addItem(ax)

        self.setLayout(vlayout)
        self.resize(800,800)
        self.show()


"""
if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    params = io_utils.parse_parameters(config)
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)
    ex  = Application(params)
    sys.exit(app.exec_())
"""
