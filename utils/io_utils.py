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

import numpy as np

def isValid(thing, d=None):
    """
    checks if 'thing' is valid. If d (a dictionary is not None) then
    check if d['thing'] is valid.
    """
    valid = False
    
    if d is not None :
        if thing not in d.keys():
            return valid 
        else :
            thing2 = d[thing]
    else :
        thing2 = thing
    
    if thing2 is not None and thing2 is not False :
        valid = True
    
    return valid

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'crappy-crystals.py', description='phase a translationally disordered crystal')
    parser.add_argument('config', type=str, \
                        help="file name of the configuration file")
    parser.add_argument('-d', '--display', action='store_true', \
                        help="display the contents of the output.h5 file and do not phase")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    return args

def parse_cmdline_args_phasing():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'phase.py', description='phase a translationally disordered crystal')
    parser.add_argument('input', type=str, \
                        help="h5 file name of the input file")
    args = parser.parse_args()
    return args


def parse_parameters(config):
    """
    Parse values from the configuration file and sets internal parameter accordingly
    The parameter dictionary is made available to both the workers and the master nodes

    The parser tries to interpret an entry in the configuration file as follows:

    - If the entry starts and ends with a single quote, it is interpreted as a string
    - If the entry is the word None, without quotes, then the entry is interpreted as NoneType
    - If the entry is the word False, without quotes, then the entry is interpreted as a boolean False
    - If the entry is the word True, without quotes, then the entry is interpreted as a boolean True
    - If non of the previous options match the content of the entry, the parser tries to interpret the entry in order as:

        - An integer number
        - A float number
        - A string

      The first choice that succeeds determines the entry type
    """

    from collections import OrderedDict
    monitor_params = OrderedDict()

    for sect in config.sections():
        monitor_params[sect]=OrderedDict()
        for op in config.options(sect):
            monitor_params[sect][op] = config.get(sect, op)
            if monitor_params[sect][op].startswith("'") and monitor_params[sect][op].endswith("'"):
                monitor_params[sect][op] = monitor_params[sect][op][1:-1]
                continue
            if monitor_params[sect][op] == 'None':
                monitor_params[sect][op] = None
                continue
            if monitor_params[sect][op] == 'False':
                monitor_params[sect][op] = False
                continue
            if monitor_params[sect][op] == 'True':
                monitor_params[sect][op] = True
                continue
            try:
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    # attempt to pass as an array of ints e.g. '1, 2, 3'
                    try :
                        l = monitor_params[sect][op].split(',')
                        monitor_params[sect][op] = np.array(l, dtype=np.int)
                        continue
                    except :
                        try :
                            l = monitor_params[sect][op].split(',')
                            monitor_params[sect][op] = np.array(l, dtype=np.float)
                            continue
                        except :
                            pass

    return monitor_params

def if_exists_del(fnam):
    import os
    # check that the directory exists and is a directory
    output_dir = os.path.split( os.path.realpath(fnam) )[0]
    if os.path.exists(output_dir) == False :
        raise ValueError('specified path does not exist: ', output_dir)
    
    if os.path.isdir(output_dir) == False :
        raise ValueError('specified path is not a path you dummy: ', output_dir)
    
    # see if it exists and if so delete it 
    # (probably dangerous but otherwise this gets really anoying for debuging)
    if os.path.exists(fnam):
        print('\n', fnam ,'file already exists, deleting the old one and making a new one')
        os.remove(fnam)

"""
Names for things:
    measured intensity  = data
    retrieved intensity = data_retrieved
    fidelity_error 
    good_pixels   
    modulus_error 
    sample_support 
    sample_support retrieved
    solid_unit
    solid_unit_init
    solid_unit_retrieved
    config_file
"""


def write_input_output_h5(fnam, **kwargs):
    """
    read a keyword list of things and write them
    (non recursive)
    
    Names for things:
        measured intensity  = data
        retrieved intensity = data_retrieved
        fidelity_error 
        good_pixels   
        modulus_error 
        sample_support 
        sample_support retrieved
        solid_unit
        solid_unit_init
        solid_unit_retrieved
        config_file
        config_file_name 
    """
    import h5py
    if_exists_del(fnam)
    
    print('\nwriting input/output file:', fnam)
    f = h5py.File(fnam, 'w')
    for key, value in kwargs.iteritems():
        if value is None :
            continue 
        if key == 'config_file' :
            print('writing config file:', key)
            g = open(value).readlines()
            h = ''
            for line in g:
                h += line
            f.create_dataset('config_file', data = np.array(h))
            f.create_dataset('config_file_name', data = np.array(value))
        else :
            print('writing:', key, value.shape, value.dtype)
            f.create_dataset(key, data = value)
    
    f.close()

def read_input_output_h5(fnam):
    """
    read a keyword list of things from the input.h5 file 
    and return a dictionary (non recursive)

    Names for things:
        measured intensity  = data
        retrieved intensity = data_retrieved
        fidelity_error 
        good_pixels   
        modulus_error 
        sample_support 
        sample_support retrieved
        solid_unit
        solid_unit retrieved
        config_file
        config_file_name 
    """
    import h5py
    
    print('\nreading input/output file:', fnam)
    f = h5py.File(fnam, 'r')
    
    kwargs = {}
    for key in f.keys():
        if key == 'config_file':
            config_file = f[key].value
            
            print('parsing the config_file...')
            # read then pass the config file
            import StringIO
            config_file = StringIO.StringIO(config_file)
            
            config = configparser.ConfigParser()
            config.readfp(config_file)
            params = parse_parameters(config)
            
            kwargs[key] = params
        else :
            print('reading:', key, end=' ')
            
            value = f[key].value
            kwargs[key] = value
            
            print(value.dtype, value.shape)
    f.close()
    return kwargs
