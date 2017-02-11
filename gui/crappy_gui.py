"""

"""
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

# import python modules using the relative directory 
# locations this way the repository can be anywhere 
root = os.path.split(os.path.abspath(__file__))[0]

root = os.path.split(root)[0]
sys.path.append(os.path.join(root, 'utils'))

import widgets

class Gui(QtGui.QTabWidget):
    def __init__(self):
        super(Gui, self).__init__()

    def initUI(self, filename):
        self.tabs = []

        self.setMovable(True)
        #self.setTabsClosable(True)

        # Show h5 list tab
        #################
        self.tabs.append( widgets.View_h5_data_widget(filename) )
        self.addTab(self.tabs[-1], "show h5 dataset")

        # Show forward model tab
        ########################
        self.tabs.append( widgets.Forward_model_widget(filename) )
        self.addTab(self.tabs[-1], "Forward model")

        # Show phase tab
        ########################
        self.tabs.append( widgets.Phase_widget(filename) )
        self.addTab(self.tabs[-1], "Phase")

def gui(filename):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
    Mwin.setWindowTitle(filename)
    
    cw = Gui()
    cw.initUI(filename)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='A gui widget container for crappy crystal stuff')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.h5 file")
    
    args = parser.parse_args()
    
    # check that h5 file exists
    if not os.path.exists(args.filename):
        outputdir = os.path.split(os.path.abspath(args.filename))[0]
        
        # mkdir if it does not exist
        if not os.path.exists(outputdir):
            yn = raw_input(str(args.filename) + ' does not exist. Create it? [y]/n : ')
            print('yn:', yn)
            if yn.strip() == 'y' or yn.strip() == '' :
                os.makedirs(outputdir)
                
                # make an empty file
                f = h5py.File(args.filename, 'w')
                f.close()
            else :
                raise NameError('h5 file does not exist: ' + args.filename)
    
    return args

if __name__ == '__main__':
    args = parse_cmdline_args()
    
    gui(args.filename)
