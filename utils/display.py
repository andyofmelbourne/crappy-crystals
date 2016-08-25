import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import signal

from io_utils import read_input_output_h5

# insert the directory in which this file is being executed from
# into sys.path
import os, sys
sys.path.append(os.path.abspath(__file__)[:-len(__file__)])

import crappy_crystals.phasing.symmetry_operations as symmetry_operations 

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


def make_crystal(fnam):
    from crappy_crystals import symmetry_operations
    import disorder
    
    # read the h5 file 
    kwargs = read_input_output_h5(fnam)

    config = kwargs['config_file']
    
    if 'solid_unit_retrieved' in kwargs.keys():
        print '\nsolid_unit = solid_unit_retrieved'
        solid_unit = kwargs['solid_unit_retrieved']
    elif 'solid_unit_init' in kwargs.keys():
        print '\nsolid_unit = solid_unit_init'
        solid_unit = kwargs['solid_unit_init']
    elif 'solid_unit' in kwargs.keys():
        print '\nsolid_unit = solid_unit'
        solid_unit = kwargs['solid_unit']
    
    if config['crystal']['space_group'] == 'P1':
        sym_ops = symmetry_operations 
        sym_ops_obj = sym_ops.P1(config['crystal']['unit_cell'], config['detector']['shape'])
    elif config['crystal']['space_group'] == 'P212121':
        sym_ops = symmetry_operations
        sym_ops_obj = sym_ops.P212121(config['crystal']['unit_cell'], config['detector']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)

    modes = sym_ops_obj.solid_syms_Fourier(Solid_unit)
    
    #N   = config['disorder']['n']
    #exp = disorder.make_exp(config['disorder']['sigma'], config['detector']['shape'])
    
    lattice = sym_ops.lattice(config['crystal']['unit_cell'], config['detector']['shape'])
    
    #diff  = N * exp * np.abs(lattice * np.sum(modes, axis=0)**2)
    #diff += (1. - exp) * np.sum(np.abs(modes)**2, axis=0)


    fourier_space_crystal = np.sum(modes, axis=0) * lattice
    real_space_crystal    = np.fft.ifftn(fourier_space_crystal)
    real_space_crystal    = np.fft.fftshift(real_space_crystal)

    return real_space_crystal

def show_crystal(fnam):
    c   = np.abs(make_crystal(fnam))**2
    c[c.real < 0] = 0
    iso = Iso_surface(np.abs(c.real))


class Iso_surface():
    def __init__(self, data, lvl = 0.1):
        from pyqtgraph.Qt import QtCore, QtGui
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl

        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.show()
        w.setWindowTitle('crystal unit cell')

        w.setCameraPosition(distance=data.shape[0])

        #g = gl.GLGridItem()
        #g.scale(2,2,2)
        #w.addItem(g)
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.scale(data.shape[0]/20.,data.shape[1]/20.,data.shape[2]/20.)
        gx.translate(-data.shape[0]/2, 0, 0)
        w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.scale(data.shape[0]/20.,data.shape[1]/20.,data.shape[2]/20.)
        gy.translate(0, -data.shape[1]/2, 0)
        w.addItem(gy)
        gz = gl.GLGridItem()
        gz.scale(data.shape[0]/20.,data.shape[1]/20.,data.shape[2]/20.)
        gz.translate(0, 0, -data.shape[1]/2)
        w.addItem(gz)
        
        import numpy as np
        
        print("Generating isosurface..")
        verts, faces = pg.isosurface(data, data.max()*lvl)
        
        md = gl.MeshData(vertexes=verts, faces=faces)
        
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[:,3] = 0.2
        colors[:,2] = np.linspace(0, 1, colors.shape[0])
        md.setFaceColors(colors)
        #m1 = gl.GLMeshItem(meshdata=md, smooth=False, shader='balloon')
        #m1.setGLOptions('additive')

        #w.addItem(m1)
        #m1.translate(-data.shape[0]/2, -data.shape[1]/2, -data.shape[2]/2)

        m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
        m2.setGLOptions('additive')

        w.addItem(m2)
        m2.translate(-data.shape[0]/2, -data.shape[1]/2, -data.shape[2]/2)
        app.exec_()

class Application():

    def __init__(self, fnam, **kwargs):
        real_space_crystal = make_crystal(fnam)
        
        if 'solid_unit_retrieved' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit_retrieved']
        elif 'solid_unit_init' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit_init']
        elif 'solid_unit' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit']
        
        #solid_unit_ret = solid_unit_ret.real
        solid_unit_ret = real_space_crystal.real
        duck_plots = (np.sum(solid_unit_ret, axis=0),\
                      np.sum(solid_unit_ret, axis=1),\
                      np.sum(solid_unit_ret, axis=2))
        
        duck_plots = np.hstack(duck_plots)
        
        if 'data_retrieved' in kwargs.keys():
            diff = kwargs['data_retrieved']
        elif 'data' in kwargs.keys():
            diff = kwargs['data']
        diff_plots = np.hstack((np.fft.ifftshift(diff[0, :, :]), \
                                np.fft.ifftshift(diff[:, 0, :]), \
                                np.fft.ifftshift(diff[:, :, 0])))
        diff_plots = diff_plots**0.2
        
        # Always start by initializing Qt (only once per application)
        app = QtGui.QApplication([])

        # Define a top-level widget to hold everything
        w = QtGui.QWidget()

        # 2D projection images for the sample
        self.duck_plots = pg.ImageView()

        # 2D slices for the diffraction volume
        self.diff_plots = pg.ImageView()

        # line plots of the error metrics
        self.plot_emod = pg.PlotWidget()
        self.plot_efid = pg.PlotWidget()
         
        Vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical) 

        # ducks
        Vsplitter.addWidget(self.duck_plots)
        
        # diffs
        Vsplitter.addWidget(self.diff_plots)
        
        # errors
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        Hsplitter.addWidget(self.plot_emod)
        Hsplitter.addWidget(self.plot_efid)
        Vsplitter.addWidget(Hsplitter)
        
        hlayout_tot = QtGui.QHBoxLayout()
        hlayout_tot.addWidget(Vsplitter)

        w.setLayout(hlayout_tot)

        self.duck_plots.setImage(duck_plots.T)
        self.diff_plots.setImage(diff_plots.T)

        if 'modulus_error' in kwargs.keys():
            emod = kwargs['modulus_error']
            if len(emod.shape) == 1 :
                self.plot_emod.plot(emod)
            else :
                for e in emod :
                    self.plot_emod.plot(e)
            self.plot_emod.setTitle('Modulus error l2norm:')
        
        """
        if 'fidelity_error' in kwargs.keys():
            efid = kwargs['fidelity_error']
            self.plot_efid.plot(efid)
            self.plot_efid.setTitle('Fidelity error l2norm:')
        """
        
        ## Display the widget as a new window
        w.show()

        
        if 'B_rav' in kwargs.keys():
            B_rav = kwargs['B_rav']
            
            # Define a top-level widget to hold everything
            w3 = QtGui.QWidget()

            # line plots of the B_rav
            self.plot_B_rav = pg.PlotWidget()

            Vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical) 

            Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
            Hsplitter.addWidget(self.plot_B_rav)
            Vsplitter.addWidget(Hsplitter)
            
            hlayout_tot = QtGui.QHBoxLayout()
            hlayout_tot.addWidget(Vsplitter)

            w3.setLayout(hlayout_tot)
            
            self.plot_B_rav.plot(B_rav)
            self.plot_B_rav.setTitle('radial average of the background')
            
            ## Display the widget as a new window
            w3.show()

        print 'showing'
        ## Start the Qt event loop

        sys.exit(app.exec_())
        
        
def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'display.py', description='display the contents of output.h5 in a GUI')
    parser.add_argument('path', type=str, \
                        help="path to output.h5 file")
    parser.add_argument('-i', '--isosurface', action='store_true', \
                        help="display the crystal in the field of view as an iso surface")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.path):
        raise NameError('output h5 file does not exist: ' + args.path)
    return args



if __name__ == '__main__':
    args = parse_cmdline_args()
    
    if args.isosurface :
        show_crystal(args.path)
    else :
        # read the h5 file 
        kwargs = read_input_output_h5(args.path)
        
        signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
        #app = QtGui.QApplication(sys.argv)
        ex  = Application(fnam = args.path, **kwargs)
