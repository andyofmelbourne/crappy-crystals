#!/usr/bin/env python

import sys
import os
import ConfigParser
import numpy as np
import subprocess

import solid_units
import utils

def generate_diff(config):
    solid_unit = solid_units.duck_3D.make_3D_duck(shape = config['simulation']['shape'])
    
    if config['simulation']['space_group'] == 'P1':
        import symmetry_operations.P1 as sym_ops
        sym_ops_obj = sym_ops.P1(config['simulation']['unit_cell'], config['detector']['shape'])
    elif config['simulation']['space_group'] == 'P212121':
        import symmetry_operations.P212121 as sym_ops
        sym_ops_obj = sym_ops.P212121(config['simulation']['unit_cell'], config['detector']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)

    modes = sym_ops_obj.solid_syms_Fourier(Solid_unit)
    
    N   = config['simulation']['n']
    exp = utils.disorder.make_exp(config['simulation']['sigma'], config['detector']['shape'])
    
    lattice = sym_ops.lattice(config['simulation']['unit_cell'], config['detector']['shape'])
    
    diff  = N * exp * np.abs(lattice * np.sum(modes, axis=0)**2)
    diff += (1. - exp) * np.sum(np.abs(modes)**2, axis=0)

    # add noise
    if config['simulation']['photons'] is not None :
        diff, edges = utils.add_noise_3d.add_noise_3d(diff, config['simulation']['photons'], \
                                      remove_courners = config['simulation']['cut_courners'],\
                                      unit_cell_size = config['simulation']['unit_cell'])
    else :
        edges = np.ones_like(diff, dtype=np.bool)

    # add gaus background
    if 'background' in config['simulation'] and config['simulation']['background'] == 'gaus':
        sig   = config['simulation']['background_std']
        scale = config['simulation']['background_scale']
        scale *= diff.max()
        print '\nAdding gaussian to diff scale (absolute), std:', scale, sig
        gaus = utils.gaus.gaus(diff.shape, scale, sig)
        diff += gaus
        background = gaus
    else :
        background = None

    # define the solid_unit support
    if config['simulation']['support_frac'] is not None :
        support = utils.padding.expand_region_by(solid_unit_expanded > 0.1, config['simulation']['support_frac'])
    else :
        support = solid_unit_expanded > (solid_unit_expanded.min() + 1.0e-5)
    
    # add a beamstop
    if config['simulation']['beamstop'] is not None :
        beamstop = utils.beamstop.make_beamstop(diff.shape, config['simulation']['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)

    print 'Simulation: number of voxels in solid unit:', np.sum(solid_unit_expanded > (1.0e-5 + solid_unit_expanded.min()))
    print 'Simulation: number of voxels in support   :', np.sum(support)


    return diff, beamstop, edges, support, solid_unit_expanded, background


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args()
    
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = utils.io_utils.parse_parameters(config)

    # display the result
    if args.display :
        script_dir = os.path.dirname(__file__)
        
        # input
        display_fnam  = os.path.join(script_dir, 'utils/display.py')
        runstr = "python " + display_fnam + " " + \
                         os.path.join(params['output']['path'],'input.h5 &')
        print '\n',runstr
        subprocess.call([runstr], shell=True)

        # crystal
        display_fnam  = os.path.join(script_dir, 'utils/display.py')
        runstr = "python " + display_fnam + " " + \
                         os.path.join(params['output']['path'],'output.h5 -i &')
        print '\n',runstr
        subprocess.call([runstr], shell=True)
        
        # output
        display_fnam  = os.path.join(script_dir, 'utils/display.py')
        runstr = "python " + display_fnam + " " + \
                         os.path.join(params['output']['path'],'output.h5')
        print '\n',runstr
        subprocess.call([runstr], shell=True)
        sys.exit()
    
    # forward problem
    if params['simulation']['sample'] == 'duck':
        diff, beamstop, edges, support, solid_unit, background = generate_diff(params)
        
        # write to file
        fnam = os.path.join(params['output']['path'], 'input.h5')
        mask = beamstop * edges
        utils.io_utils.write_input_output_h5(fnam, data = diff, sample_support = support, \
                good_pix = mask, solid_unit = solid_unit, background = background,\
                config_file = args.config)

    # inverse problem
    runstr = "python " + params['phasing']['script'] + ' ' + \
                     os.path.join(params['output']['path'],'input.h5')
    print '\n',runstr
    subprocess.call([runstr], shell=True)

