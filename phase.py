#!/usr/bin/env python

import sys
import os.path
import ConfigParser
import numpy as np

sys.path.append('.')
from phasing.phase       import phase
from solid_units.duck_3D import make_3D_duck
from utils.disorder      import make_exp
from utils.io_utils      import parse_parameters
from utils.io_utils      import parse_cmdline_args

def generate_diff(config):
    solid_unit = make_3D_duck(shape = config['solid_unit']['shape'])
    
    if config['crystal']['space_group'] == 'P1':
        import symmetry_operations.P1 as sym_ops 

    unit_cell = sym_ops.unit_cell(solid_unit, config['crystal']['unit_cell'])
    Unit_cell = np.fft.fftn(unit_cell, config['detector']['shape'])

    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])

    N   = config['disorder']['n']
    exp = make_exp(config['disorder']['sigma'], config['detector']['shape'])

    lattice = sym_ops.lattice(config['crystal']['unit_cell'], config['detector']['shape'])
    
    diff  = N * exp * np.abs(lattice * Unit_cell)**2 
    diff += (1. - exp) * np.abs(Solid_unit)**2 
    return diff



if __name__ == "__main__":
    args = parse_cmdline_args()
    
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = parse_parameters(config)

    # forward problem
    diff = generate_diff(params)

    # inverse problem
    phase(diff, params)
