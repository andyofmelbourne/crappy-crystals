#!/usr/bin/env python

# for python 2 / 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try :
    range = xrange
except NameError :
    pass

try :
    import ConfigParser as configparser 
except ImportError :
    import configparser 

import pyqtgraph as pg
try :
  from PyQt5 import QtGui, QtCore
except ImportError :
  from PyQt4 import QtGui, QtCore

import signal

import numpy as np
import h5py 
import argparse
import os, sys
import copy

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]

root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'gui/h5-viewer'))

from h5_viewer import View_h5_data_widget
import io_utils

def load_config(filename, name = 'basic_stitch.ini'):
    """
    Read the config file 'name' from the same directory
    as filename. If it does not exist then try to copy it
    from 'root/process/name' to 'filename's directory. If
    it does not exist in the process folder then throw an 
    error. Finally parse the config file and return the 
    dictionary.
    """
    # if config is non then read the default from the *.pty dir
    dirnam = os.path.split(filename)[0]
    config = os.path.join(dirnam, name)
    if not os.path.exists(config):
        config = os.path.join(root, 'process')
        config = os.path.join(config, name)
         
        # check that args.config exists
        if not os.path.exists(config):
            raise NameError('config file does not exist: ' + config)

        # copy it to the filename dir.
        try :
            import shutil
            shutil.copy(config, dirnam)
        except Exception as e :
            print(e)
    
    # process config file
    conf = configparser.ConfigParser()
    conf.read(config)
    
    params = io_utils.parse_parameters(conf)
    return params


class Show_h5_list_widget(QtGui.QWidget):
    def __init__(self, filename, names = None):
        super(Show_h5_list_widget, self).__init__()

        self.filename = filename
        self.names    = names
        
        # add the names to Qlist thing
        self.listWidget = QtGui.QListWidget(self)
        #self.listWidget.setMinimumWidth(self.listWidget.sizeHintForColumn(0))
        #self.listWidget.setMinimumHeight(500)
        
        # update list button
        ####################
        self.update_button = QtGui.QPushButton('update', self)
        self.update_button.clicked.connect(self.update)

        # get the list of groups and items
        self.dataset_names = [] 
        self.dataset_items = [] 
        
        f = h5py.File(filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()

        self.initUI()
    
    def initUI(self):
        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.listWidget)
        layout.addWidget(self.update_button)
        
        # add the layout to the central widget
        self.setLayout(layout)

    def add_dataset_name(self, name, obj):
        names = self.names
        if isinstance(obj, h5py.Dataset):
            if ((names is None) or (names is not None and name in names)) \
                    and name not in self.dataset_names:
                self.dataset_names.append(name)
                self.dataset_items.append(QtGui.QListWidgetItem(self.listWidget))
                self.dataset_items[-1].setText(name)
    
    def update(self):
        f = h5py.File(self.filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()


class Show_nd_data_widget(QtGui.QWidget):
    def __init__(self):
        super(Show_nd_data_widget, self).__init__()

        self.plotW  = None
        self.plotW2 = None
        self.layout = None
        self.name   = None
        self.initUI()
    
    def initUI(self):
        # set the layout
        self.layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(self.layout)
    
    def show(self, filename, name, refresh=False):
        """
        plots:
            (N,)      float, int          --> line plot
            (N, M<4)  float, int          --> line plots
            (N, M>4)  float, complex, int --> 2d image
            (N, M>4)  complex             --> 2d images (abs, angle, real, imag)
            (N, M, L) float, complex, int --> 2d images (real) with slider
        """
        # make plot
        f = h5py.File(filename, 'r')
        shape = f[name].shape

        if len(shape) == 1 :
            if refresh :
                self.plotW.setData(f[name][()])
            else :
                self.plotW = pg.PlotWidget(title = name)
                self.plotW.plot(f[name][()], pen=(255, 150, 150))
        
        elif len(shape) == 2 and shape[1] < 4 :
            pens = [(255, 150, 150), (150, 255, 150), (150, 150, 255)]
            if refresh :
                self.plotW.clear()
                for i in range(shape[1]):
                    self.plotW.setData(f[name][:, i], pen=pens[i])
            else :
                self.plotW = pg.PlotWidget(title = name + ' [0, 1, 2] are [r, g, b]')
                for i in range(shape[1]):
                    self.plotW.plot(f[name][:, i], pen=pens[i])

        elif len(shape) == 2 :
            if refresh :
                self.plotW.setImage(f[name][()].real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
            else :
                if 'complex' in f[name].dtype.name :
                    title = name + ' (abs, angle, real, imag)'
                else :
                    title = name
                
                frame_plt = pg.PlotItem(title = title)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                if 'complex' in f[name].dtype.name :
                    im = f[name][()].T
                    self.plotW.setImage(np.array([np.abs(im), np.angle(im), im.real, im.imag]))
                else :
                    self.plotW.setImage(f[name][()].T)

        elif len(shape) == 3 :
            if refresh :
                replot_frame()
            else :
                # show the first frame
                frame_plt = pg.PlotItem(title = name)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                self.plotW.setImage(f[name][0].real.T)
                
                # add a little 1d plot with a vline
                self.plotW2 = pg.PlotWidget(title = 'index')
                self.plotW2.plot(np.arange(f[name].shape[0]), pen=(255, 150, 150))
                vline = self.plotW2.addLine(x = 0, movable=True, bounds = [0, f[name].shape[0]-1])
                self.plotW2.setMaximumSize(10000000, 100)
                
                def replot_frame():
                    i = int(vline.value())
                    f = h5py.File(filename, 'r')
                    self.plotW.setImage( f[name][i].real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
                    f.close()
                    
                vline.sigPositionChanged.connect(replot_frame)

        f.close()
         
        # add to layout
        self.layout.addWidget(self.plotW, stretch = 1)
        
        if self.plotW2 is not None :
            self.layout.addWidget(self.plotW2, stretch = 0)
        
        # remember the last file and dataset for updating
        self.name     = name
        self.filename = filename
    
    def close(self):
        # remove from layout
        if self.layout is not None :
            if self.plotW is not None :
                self.layout.removeWidget(self.plotW)
            
            if self.plotW2 is not None :
                self.layout.removeWidget(self.plotW2)
        
        # close plot widget
        if self.plotW is not None :
            self.plotW.close()
            self.plotW = None
        
        if self.plotW2 is not None :
            self.plotW2.close()
            self.plotW2 = None
    
    def update(self):
        # update the current plot
        self.show(self.filename, self.name, True)



class Test_run_command_widget(QtGui.QWidget):
    def __init__(self, h5_filename):
        super(Test_run_command_widget, self).__init__()

        self.h5_filename = h5_filename
        
        # set the python filename
        pyname = 'template_command.py'
        
        self.py = os.path.join(root, 'process/' + pyname)
        
        # read the config file
        self.config_dict  = load_config(h5_filename, name = pyname[:-2] + 'ini')
        self.output_dir   = os.path.split(h5_filename)[0]
        self.config_out   = os.path.join(self.output_dir, pyname[:-2] + 'ini')
        self.config_group = 'template_command'
        
        self.cmd = 'python ' + self.py + ' ' + h5_filename + ' -c ' + self.config_out
        
        self.run_com_Widget = Run_command_template_widget(\
                             h5_filename, self.config_dict, self.config_out, \
                             self.config_group, self.cmd, h5_datas = ['image', 'error'])
        self.initUI()
    
    def initUI(self):
        # Make a grid layout
        layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        layout.addWidget(self.run_com_Widget)

        
class Run_command_template_widget(QtGui.QWidget):
    """
    I take a h5 filename and a config dictionary. You should
    subclass me. 
    
    GUI layout:

    Widget box:         output tabs
    Run command button  frame / error plots ...
    ...

    Config editor:
    output_group
    number of frames
    ...
    
    Status:
    Command:
    """
    def __init__(self, h5_filename, config_dict, config_out, config_group \
                 ,cmd, h5_datas = ['image', 'error']):
        super(Run_command_template_widget, self).__init__()
        
        self.h5_filename = h5_filename
        self.config_dict = config_dict
        self.config_out  = config_out
        self.config_group = config_group
        self.h5_datas    = h5_datas
        self.cmd         = cmd
        
        self.h5_out       = self.config_dict[config_group]['output_file']
        self.h5_out_group = self.config_dict[config_group]['output_group']
        if self.h5_out is None :
            self.h5_out = h5_filename
        
        self.initUI()

    def initUI(self):
        """
        """
        # Make a grid layout
        layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        # view data widget
        ##################
        # make a list of the output to look out for 
        out_names = [self.h5_out_group + '/' + d for d in self.h5_datas]
        self.view_output_widget = View_h5_data_widget(self.h5_out, names = out_names)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(self.config_dict, self.config_out)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # A do something button
        ##################
        self.do_button = QtGui.QPushButton('do something', self)
        self.do_button.clicked.connect(self.do_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        vlay = QtGui.QVBoxLayout() 
        vlay.addWidget(self.run_button)
        vlay.addWidget(self.do_button)
        vlay.addWidget(self.config_widget)
        vlay.addStretch(1)
        
        hlay = QtGui.QHBoxLayout() 
        hlay.addLayout(vlay, stretch = 0)
        hlay.addWidget(self.view_output_widget, stretch = 1)

        layout.addLayout(hlay)
        layout.addWidget(self.run_command_widget)

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
          
        # Run the command 
        #################
        self.run_command_widget.run_cmd(self.cmd)
    
    def finished(self):
        self.view_output_widget.update()
    
    def do_button_clicked(self):
        print('you pressed the do button: I do nothing')


class Write_config_file_widget(QtGui.QWidget):
    def __init__(self, config_dict, output_filename):
        super(Write_config_file_widget, self).__init__()
        
        self.output_filename = output_filename
        self.initUI(config_dict)
    
    def initUI(self, config_dict):
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        self.output_config = copy.deepcopy(config_dict)
        
        i = 0
        # add the output config filename 
        ################################    
        fnam_label = QtGui.QLabel(self)
        fnam_label.setText(self.output_filename)
        
        # add the label to the layout
        layout.addWidget(fnam_label, i, 0, 1, 2)
        i += 1
        
        # we have 
        self.labels_lineedits = {}
        group_labels = []
        for group in config_dict.keys():
            # add a label for the group
            group_labels.append( QtGui.QLabel(self) )
            group_labels[-1].setText(group)
            # add the label to the layout
            layout.addWidget(group_labels[-1], i, 0, 1, 2)
            i += 1
            
            self.labels_lineedits[group] = {}
            # add the labels and line edits
            for key in config_dict[group].keys():
                self.labels_lineedits[group][key] = {}
                self.labels_lineedits[group][key]['label'] = QtGui.QLabel(self)
                self.labels_lineedits[group][key]['label'].setText(key)
                layout.addWidget(self.labels_lineedits[group][key]['label'], i, 0, 1, 1)
                
                self.labels_lineedits[group][key]['lineedit'] = QtGui.QLineEdit(self)
                # special case when config_dict[group][key] is a list
                if type(config_dict[group][key]) is list or type(config_dict[group][key]) is np.ndarray :
                    setT = ''
                    for j in range(len(config_dict[group][key])-1):
                        setT += str(config_dict[group][key][j]) + ','
                    setT += str(config_dict[group][key][-1])
                else :
                    setT = str(config_dict[group][key])
                self.labels_lineedits[group][key]['lineedit'].setText(setT)
                self.labels_lineedits[group][key]['lineedit'].editingFinished.connect(self.write_file)
                layout.addWidget(self.labels_lineedits[group][key]['lineedit'], i, 1, 1, 1)
                i += 1

    def write_file(self):
        with open(self.output_filename, 'w') as f:
            for group in self.labels_lineedits.keys():
                f.write('['+group+']' + '\n')
                
                for key in self.labels_lineedits[group].keys():
                    out_str = key
                    out_str = out_str + ' = '
                    out_str = out_str + str(self.labels_lineedits[group][key]['lineedit'].text())
                    f.write( out_str + '\n')


class Forward_model_widget(QtGui.QWidget):
    def __init__(self, filename):
        super(Forward_model_widget, self).__init__()
        
        config_dict = load_config(filename, name = 'forward_model.ini')
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'forward_model.ini')
        self.filename = filename
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # plots
        #######
        self.crystal_path = '/forward_model/crystal'
        self.diff_path = '/forward_model/data'
        self.displayW = self.display(init=True)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(lambda : self.display(init=False))
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate forward model', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.displayW,            0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/forward_model.py')
        cmd = 'python ' + py + ' -f ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def display(self, init=False):
        self.f = h5py.File(self.filename, 'r')
        print(self.crystal_path, init)
        
        if init :
            # crystal
            frame_plt = pg.PlotItem(title = 'real space crystal projections')
            self.imageView = pg.ImageView(view = frame_plt)
            self.imageView.ui.menuBtn.hide()
            self.imageView.ui.roiBtn.hide()
            
            # diff
            frame_plt2 = pg.PlotItem(title = 'diffraction volume slices')
            self.imageView2 = pg.ImageView(view = frame_plt2)
            self.imageView2.ui.menuBtn.hide()
            self.imageView2.ui.roiBtn.hide()
            
            self.im_init = False
    
            splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
            splitter.addWidget(self.imageView)
            splitter.addWidget(self.imageView2)
        
        if self.crystal_path in self.f :
            print('making arrays...')

            # real-space crystal view 
            #########################
            #cryst = self.f['/forward_model/solid_unit'][()].real
            cryst = np.abs(self.f[self.crystal_path][()])
            #cryst = np.fft.fftshift(cryst)
            # add 10 pix padding
            padd = np.zeros((10, cryst.shape[0]), dtype=cryst.real.dtype)
            t = (np.sum(cryst,axis=0), padd, np.sum(cryst,axis=1), padd, np.sum(cryst,axis=2))
            #cryst = reduce(np.multiply.outer, [np.ones((128,)), np.ones((128,)),np.arange(128)])
            t = (np.sum(cryst,axis=0), padd, np.sum(cryst,axis=1), padd, np.sum(cryst,axis=2))
            t = np.concatenate(t, axis=0)
            
            # diffraction volume view 
            #########################
            diff_h5 = self.f[self.diff_path]
            
            diff = [np.fft.fftshift(diff_h5[0])[:, ::-1], np.fft.fftshift(diff_h5[:, 0, :])[:,::-1], np.fft.fftshift(diff_h5[:, :, 0])[:,::-1]]
            tt = (diff[0], padd, diff[1], padd, diff[2])
            tt = np.concatenate(tt, axis=0)**0.2
             
            if init is False :
                print('updating image, init = False')
                self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.imageView2.setImage(tt, autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.im_init = True
            
            else :
                print('updating image, init = True')
                self.imageView.setImage(t)
                self.imageView2.setImage(tt)
                self.im_init = True
                    
        self.f.close()

        if init :
            return splitter

class Phase_widget(QtGui.QWidget):
    def __init__(self, filename):
        super(Phase_widget, self).__init__()
        
        config_dict = load_config(filename, name = 'phase.ini')
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'phase.ini')
        self.filename = filename
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # plots
        #######
        self.crystal_path = '/phase/crystal'
        self.diff_path    = '/phase/diff'
        self.displayW     = self.display(init=True)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(lambda : self.display(init=False))
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('phase', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.displayW,            0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/phase.py')
        cmd = 'python ' + py + ' -f ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def display(self, init=False):
        self.f = h5py.File(self.filename, 'r')
        print(self.crystal_path, init)
        
        if init :
            # crystal
            frame_plt = pg.PlotItem(title = 'real space crystal projections')
            self.imageView = pg.ImageView(view = frame_plt)
            self.imageView.ui.menuBtn.hide()
            self.imageView.ui.roiBtn.hide()
            
            # diff
            frame_plt2 = pg.PlotItem(title = 'diffraction volume slices')
            self.imageView2 = pg.ImageView(view = frame_plt2)
            self.imageView2.ui.menuBtn.hide()
            self.imageView2.ui.roiBtn.hide()
            
            self.im_init = False
             
            splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
            splitter.addWidget(self.imageView)
            splitter.addWidget(self.imageView2)
        
        if self.crystal_path in self.f :
            print('making arrays...')

            # real-space crystal view 
            #########################
            #cryst = self.f['/forward_model/solid_unit'][()].real
            cryst = np.abs(self.f[self.crystal_path][()])
            #cryst = np.fft.fftshift(cryst)
            # add 10 pix padding
            padd = np.zeros((10, cryst.shape[0]), dtype=cryst.real.dtype)
            t = (np.sum(cryst,axis=0), padd, np.sum(cryst,axis=1), padd, np.sum(cryst,axis=2))
            #cryst = reduce(np.multiply.outer, [np.ones((128,)), np.ones((128,)),np.arange(128)])
            t = (np.sum(cryst,axis=0), padd, np.sum(cryst,axis=1), padd, np.sum(cryst,axis=2))
            t = np.concatenate(t, axis=0)
            
            # diffraction volume view 
            #########################
            diff_h5 = self.f[self.diff_path]
            
            diff = [np.fft.fftshift(diff_h5[0])[:, ::-1], np.fft.fftshift(diff_h5[:, 0, :])[:,::-1], np.fft.fftshift(diff_h5[:, :, 0])[:,::-1]]
            tt = (diff[0], padd, diff[1], padd, diff[2])
            tt = np.concatenate(tt, axis=0)**0.2
             
            if init is False :
                print('updating image, init = False')
                self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.imageView2.setImage(tt, autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.im_init = True
            
            else :
                print('updating image, init = True')
                self.imageView.setImage(t)
                self.imageView2.setImage(tt)
                self.im_init = True
                    
        self.f.close()

        if init :
            return splitter

class Run_and_log_command(QtGui.QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super(Run_and_log_command, self).__init__()
        
        self.polling_interval = 0.1
        self.initUI()
        
    def initUI(self):
        """
        Just setup a qlabel showing the shell command
        and another showing the status of the process
        """
        # Make a grid layout
        #layout = QtGui.QGridLayout()
        hbox = QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(hbox)
        
        # show the command being executed
        self.command_label0 = QtGui.QLabel(self)
        self.command_label0.setText('<b>Command:</b>')
        self.command_label  = QtGui.QLabel(self)
        #self.command_label.setMaximumSize(50, 250)
         
        # show the status of the command
        self.status_label0  = QtGui.QLabel(self)
        self.status_label0.setText('<b>Status:</b>')
        self.status_label   = QtGui.QLabel(self)
        
        # add to layout
        hbox.addWidget(self.status_label0)
        hbox.addWidget(self.status_label)
        hbox.addWidget(self.command_label0)
        hbox.addWidget(self.command_label)
        hbox.addStretch(1)

        #layout.addWidget(self.status_label0,  0, 0, 1, 1)
        #layout.addWidget(self.status_label,   0, 1, 1, 1)
        #layout.addWidget(self.command_label0, 1, 0, 1, 1)
        #layout.addWidget(self.command_label,  1, 1, 1, 1)
         
    def run_cmd(self, cmd):
        from subprocess import PIPE, Popen
        import shlex
        self.command_label.setText(cmd)
        self.status_label.setText('running the command')
        self.p = Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
        
        # start a Qt timer to update the status
        QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
    
    def update_status(self):
        status = self.p.poll()
        if status is None :
            self.status_label.setText('Running')
             
            # start a Qt timer to update the status
            QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
        
        elif status is 0 :
            self.status_label.setText('Finished')
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            
            # emmit a signal when complete
            self.finished_signal.emit(True)
            print('Output   :', self.output.decode("utf-8"))
            
        else :
            self.status_label.setText(str(status))
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            print('Output   :', self.output.decode("utf-8"))
            print('Error msg:', self.err_msg.decode("utf-8"))
            
            # emmit a signal when complete
            self.finished_signal.emit(False)


def run_and_log_command():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
    Mwin.setWindowTitle('run and log command')
    
    cw = Run_and_log_command()
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    print('running command')
    import time
    time.sleep(1)
    cw.run_cmd('mpirun -n 4 python test.py')

    def print_finished(x):
        if x :
            print('Finished!')
        else :
            print('Something went wrong...')

    cw.finished_signal.connect(print_finished)

    print('app exec')
    Mwin.show()
    app.exec_()


if __name__ == '__main__':
    run_and_log_command()
