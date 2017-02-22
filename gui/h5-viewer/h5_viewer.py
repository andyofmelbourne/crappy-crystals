#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
import numpy as np
import h5py
#import scipy.constants as sc

import pyqtgraph as pg
try:
  from PyQt5 import QtGui, QtCore
except ImportError:
  from PyQt4 import QtGui, QtCore

import signal

#import copy 

#import ConfigParser


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

        
        if shape == () :
            if refresh :
                self.plotW.setData(f[name][()])
            else :
                self.plotW = self.text_label = QtGui.QLabel(self)
                self.plotW.setText('<b>'+name+'</b>: ' + str(f[name][()]))

        elif len(shape) == 1 :
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


class View_h5_data_widget(QtGui.QWidget):
    def __init__(self, filename, names = None):
        super(View_h5_data_widget, self).__init__()
        
        self.filename = filename
        self.names = names
            
        self.show_list_widget = Show_h5_list_widget(filename, names = names)
        self.plot1dWidget = Show_nd_data_widget()
        
        # send a signal when an item is clicked
        self.show_list_widget.listWidget.itemClicked.connect(self.dataset_clicked)

        self.initUI()

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        # add the h5 datasets list
        layout.addWidget(self.show_list_widget)
        
        # add the 1d viewer 
        layout.addWidget(self.plot1dWidget, stretch=1)
        

    def dataset_clicked(self, item):
        name = str(item.text())
        
        # close the last image
        self.plot1dWidget.close()
        
        # load the new one
        self.plot1dWidget.show(self.filename, name)
        
    def update(self):
        self.show_list_widget.update()
        self.plot1dWidget.update()

def gui(filename):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
    Mwin.setWindowTitle(filename)
    
    cw = View_h5_data_widget(filename)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

if __name__ == '__main__':
    gui(sys.argv[1])
