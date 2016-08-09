#!/usr/bin/env python

import sys
import os
import ConfigParser
import numpy as np
import subprocess

import crappy_crystals
import crappy_crystals.utils as utils

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
        diff, beamstop, edges, support, solid_unit, background = utils.generate_diff(params)
        
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

