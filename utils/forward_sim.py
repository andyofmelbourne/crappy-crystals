import numpy as np
import crappy_crystals.solid_units 
import crappy_crystals.phasing.symmetry_operations as symmetry_operations 
import disorder


def generate_diff(config):
    solid_unit = crappy_crystals.solid_units.duck_3D.make_3D_duck(shape = config['simulation']['shape'])
    
    if config['simulation']['space_group'] == 'P1':
        sym_ops = symmetry_operations.P1(config['simulation']['unit_cell'], config['detector']['shape'])
    elif config['simulation']['space_group'] == 'P212121':
        sym_ops = symmetry_operations.P212121(config['simulation']['unit_cell'], config['detector']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)

    # define the solid_unit support
    if config['simulation']['support_frac'] is not None :
        support = padding.expand_region_by(solid_unit_expanded > 0.1, config['simulation']['support_frac'])
    else :
        support = solid_unit_expanded > (solid_unit_expanded.min() + 1.0e-5)
    
    #solid_unit_expanded = np.random.random(support.shape)*support + 0J
    solid_unit_expanded = solid_unit_expanded * support 
    Solid_unit = np.fft.fftn(solid_unit_expanded)

    modes = sym_ops.solid_syms_Fourier(Solid_unit)
    
    N   = config['simulation']['n']
    exp = disorder.make_exp(config['simulation']['sigma'], config['detector']['shape'])
    
    lattice = symmetry_operations.lattice(config['simulation']['unit_cell'], config['detector']['shape'])
    
    diff  = N * exp * lattice * np.abs(np.sum(modes, axis=0)**2)
    diff += (1. - exp) * np.sum(np.abs(modes)**2, axis=0)

    # add noise
    if config['simulation']['photons'] is not None :
        diff, edges = add_noise_3d.add_noise_3d(diff, config['simulation']['photons'], \
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
        gaus = gaus.gaus(diff.shape, scale, sig)
        diff += gaus
        background = gaus
    else :
        background = None


    # add a beamstop
    if config['simulation']['beamstop'] is not None :
        beamstop = beamstop.make_beamstop(diff.shape, config['simulation']['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)

    print 'Simulation: number of voxels in solid unit:', np.sum(solid_unit_expanded > (1.0e-5 + solid_unit_expanded.min()))
    print 'Simulation: number of voxels in support   :', np.sum(support)


    return diff, beamstop, edges, support, solid_unit_expanded, background
