# I need a map of the molecule's electron density (not the unit-cell's)
# as one continuous block. This volume can, of course extend beyond the unit-cell bounds.


import argparse
import sys

def parse_cmdline_args():
    parser = argparse.ArgumentParser(description="generate a molecules 3D electron density from a pdb id")
    parser.add_argument('pdbid', type=str, \
                        help="the pdb id e.g.: 5JDK")
    
    args = parser.parse_args()
    
    return args

def get_pdb_mtz(pdbid, dirnam = './temp'):
    # fetch the pdb info
    from subprocess import PIPE, Popen
    import shlex
    import os
    dirnam2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), dirnam)
    cmd = 'mkdir ' + dirnam2
    p = Popen(shlex.split(cmd))
    p.wait()
    
    cmd = 'phenix.fetch_pdb --mtz ' + pdbid
    p = Popen(shlex.split(cmd), cwd=dirnam2)
    p.wait()

if __name__ == '__main__':
    args = parse_cmdline_args()
    
    # get the *.pdb and *.mtz files in ./temp
    get_pdb_mtz(args.pdbid)
    
    # now calculate the map


