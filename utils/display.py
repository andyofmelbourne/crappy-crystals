import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import signal
import sys

from io_utils import read_input_output_h5

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

def iso_surface_test():
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('pyqtgraph example: GLIsosurface')

    w.setCameraPosition(distance=40)

    g = gl.GLGridItem()
    g.scale(2,2,1)
    w.addItem(g)

    import numpy as np

    ## Define a scalar field from which we will generate an isosurface
    def psi(i, j, k, offset=(25, 25, 50)):
        x = i-offset[0]
        y = j-offset[1]
        z = k-offset[2]
        th = np.arctan2(z, (x**2+y**2)**0.5)
        phi = np.arctan2(y, x)
        r = (x**2 + y**2 + z **2)**0.5
        a0 = 1
        #ps = (1./81.) * (2./np.pi)**0.5 * (1./a0)**(3/2) * (6 - r/a0) * (r/a0) * np.exp(-r/(3*a0)) * np.cos(th)
        ps = (1./81.) * 1./(6.*np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * np.exp(-r/(3*a0)) * (3 * np.cos(th)**2 - 1)
        
        return ps
        
        #return ((1./81.) * (1./np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * (r/a0) * np.exp(-r/(3*a0)) * np.sin(th) * np.cos(th) * np.exp(2 * 1j * phi))**2 
    
    
    print("Generating scalar field..")
    data = np.abs(np.fromfunction(psi, (50,50,100)))
    
    print("Generating isosurface..")
    verts, faces = pg.isosurface(data, data.max()/4.)

    md = gl.MeshData(vertexes=verts, faces=faces)

    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:,3] = 0.2
    colors[:,2] = np.linspace(0, 1, colors.shape[0])
    md.setFaceColors(colors)
    m1 = gl.GLMeshItem(meshdata=md, smooth=False, shader='balloon')
    m1.setGLOptions('additive')

    #w.addItem(m1)
    m1.translate(-25, -25, -20)

    m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
    m2.setGLOptions('additive')

    w.addItem(m2)
    m2.translate(-25, -25, -50)
    app.exec_()

def show_crystal(fnam):
    # read the h5 file 
    kwargs = read_input_output_h5(args.apth)
    if 'solid_unit_retrieved' in kwargs.keys():
        params = kwargs['s']

    # generate brag Fourier components
    solid_unit = solid_units.duck_3D.make_3D_duck(shape = config['solid_unit']['shape'])
    
    if config['crystal']['space_group'] == 'P1':
        import symmetry_operations.P1 as sym_ops
        sym_ops_obj = sym_ops.P1(params['crystal']['unit_cell'], params['detector']['shape'])
    elif config['crystal']['space_group'] == 'P212121':
        import symmetry_operations.P212121 as sym_ops
        sym_ops_obj = sym_ops.P212121(params['crystal']['unit_cell'], params['detector']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)

    modes = sym_ops_obj.solid_syms_Fourier(Solid_unit)
    
    N   = config['disorder']['n']
    exp = utils.disorder.make_exp(config['disorder']['sigma'], config['detector']['shape'])
    
    lattice = sym_ops.lattice(config['crystal']['unit_cell'], config['detector']['shape'])
    
    diff  = N * exp * np.abs(lattice * np.sum(modes, axis=0)**2)
    diff += (1. - exp) * np.sum(np.abs(modes)**2, axis=0)

    # add noise
    if config['detector']['photons'] is not None :
        diff, edges = utils.add_noise_3d.add_noise_3d(diff, config['detector']['photons'], \
                                      remove_courners = config['detector']['cut_courners'],\
                                      unit_cell_size = config['crystal']['unit_cell'])
    else :
        edges = np.ones_like(diff, dtype=np.bool)

    # define the solid_unit support
    if config['solid_unit']['support_frac'] is not None :
        support = utils.padding.expand_region_by(solid_unit_expanded > 0.1, config['solid_unit']['support_frac'])
    else :
        support = solid_unit_expanded > (solid_unit_expanded.min() + 1.0e-5)
    
    # add a beamstop
    if config['detector']['beamstop'] is not None :
        beamstop = utils.beamstop.make_beamstop(diff.shape, config['detector']['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)



class Iso_surface():
    def __init__(self):
        pass

class Application():

    def __init__(self, **kwargs):
        
        if 'solid_unit_retrieved' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit_retrieved']
        elif 'solid_unit_init' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit_init']
        elif 'solid_unit' in kwargs.keys():
            solid_unit_ret = kwargs['solid_unit']
        
        solid_unit_ret = np.fft.ifftshift(solid_unit_ret.real)
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
            self.plot_emod.plot(emod)
            self.plot_emod.setTitle('Modulus error l2norm:')
        
        if 'fidelity_error' in kwargs.keys():
            efid = kwargs['fidelity_error']
            self.plot_efid.plot(efid)
            self.plot_efid.setTitle('Fidelity error l2norm:')
        
        ## Display the widget as a new window
        w.show()

        ## Start the Qt event loop
        sys.exit(app.exec_())
        
        
def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'display.py', description='display the contents of output.h5 in a GUI')
    parser.add_argument('path', type=str, \
                        help="path to output.h5 file")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.path):
        raise NameError('output h5 file does not exist: ' + args.path)
    return args



if __name__ == '__main__':
    args = parse_cmdline_args()
    
    # read the h5 file 
    kwargs = read_input_output_h5(args.path)
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)
    
    ex  = Application(**kwargs)
