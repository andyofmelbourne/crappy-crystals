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
from utils.add_noise_3d  import add_noise_3d
from utils.padding       import expand_region_by
from utils.beamstop      import make_beamstop

def generate_diff(config):
    solid_unit = make_3D_duck(shape = config['solid_unit']['shape'])
    
    if config['crystal']['space_group'] == 'P1':
        import symmetry_operations.P1 as sym_ops 
    elif config['crystal']['space_group'] == 'P212121':
        import symmetry_operations.P212121 as sym_ops 
    
    unit_cell = sym_ops.unit_cell(solid_unit, config['crystal']['unit_cell'])
    Unit_cell = np.fft.fftn(unit_cell, config['detector']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)
    
    N   = config['disorder']['n']
    exp = make_exp(config['disorder']['sigma'], config['detector']['shape'])
    
    lattice = sym_ops.lattice(config['crystal']['unit_cell'], config['detector']['shape'])
    
    diff  = N * exp * np.abs(lattice * Unit_cell)**2 
    diff += (1. - exp) * np.abs(Solid_unit)**2 

    # add noise
    if config['detector']['photons'] is not None :
        diff, edges = add_noise_3d(diff, config['detector']['photons'], \
                                      remove_courners = config['detector']['cut_courners'])
    else :
        edges = np.ones_like(diff, dtype=np.bool)

    # define the solid_unit support
    if config['solid_unit']['support_frac'] is not None :
        support = expand_region_by(solid_unit_expanded > 0.1, config['solid_unit']['support_frac'])
    else :
        support = solid_unit_expanded > (solid_unit_expanded.min() + 1.0e-5)
    
    # add a beamstop
    if config['detector']['beamstop'] is not None :
        beamstop = make_beamstop(diff.shape, config['detector']['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)

    return diff, beamstop, edges, support, solid_unit_expanded



if __name__ == "__main__":
    args = parse_cmdline_args()
    
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = parse_parameters(config)

    # forward problem
    diff, beamstop, edges, support, solid_unit = generate_diff(params)

    # inverse problem
    solid_ret, diff_ret = phase(diff, support, params, \
                                good_pix = beamstop + edges, solid_known = solid_unit)

