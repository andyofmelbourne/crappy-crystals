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

import numpy as np

import symmetry_operations 
import padding
import add_noise_3d
import io_utils

def make_exp(sigma, shape):
    # make the B-factor thing
    i, j, k = np.meshgrid(np.fft.fftfreq(shape[0], 1.), \
                          np.fft.fftfreq(shape[1], 1.), \
                          np.fft.fftfreq(shape[2], 1.), indexing='ij')

    exp     = np.exp(-4. * sigma**2 * np.pi**2 * (i**2 + j**2 + k**2))
    return exp

def generate_diff(solid_unit, unit_cell, N, sigma, **params):
    """
    Generates the 3D diffraction volume of a translationally disordered crystal.
    
    The model for the 3D diffraction intensities is:
        diff = Counting_Noise(N exp lattice |\sum_i F_i|^2 
                                + (1 - exp) \sum_i |F_i|^2)
              * edges * beamstop 
    where F_i are the reciprocal solid units in the unit-cell,
    exp = exp{-4 * \sigma^2 * \pi^2 * q^2}, edges and beamstop are masks 
    (with 1 = good and 0 = bad) and Counting_Noise is a poisson random variable.
        
    Parameters
    ----------
    solid_unit : numpy.ndarray
    
    unit_cell : sequence of length 3, int
        The pixel dimensions of the unit-cell
    
    N : int
        The number of unit-cells per crystal
    
    sigma : float
        The disorder parameter of the crystal
    
    Keyword Arguments
    -----------------
    space_group : string, optional, default ('P1')
        Space group of the crystal, can only be one of:
            'P1', 'P212121'
    
    photons : integer or None or False, optional, default (None)
        The total number of photons recorded in the outer resolution shell 
        of the diffraction volume. If 'None' or 'False' then no photon 
        counting noise is added to the output diffraction volume.
    
    cut_courners : True or False, optional, default (False)
        If 'True' then the output diffraction volume is masked so that the
        courners of the cube are zero. The non-zero diffraction volume will
        then look like a sphere.
    
    beamstop : number or None or False, optional, default (None)
        If 'number' then the diffraction volume is masked at low diffraction 
        angles by a sphere of radius 'number' pixels.
    
    support_frac : number or None or False, optional, default (None)
        Padd the non-zero solid_unit pixels with a gaussian until the sample 
        support has increased by the fraction 'support_frac'.
    
    background : True or None or False, optional, default (None)
        If 'True' then a Gaussian background is added to the diffraction 
        volume. This is not Gaussian noise but an actual large Gaussian 
        of width 'background_std' pixels and scaled by 'background_scale' 
        times the maximum value of the crystal diffraction. 
    
    background_std : float or None or False, optional, default (1.)
        The standard deviation in pixels of the background. If 'background' 
        is None or False then this is ignored.
    
    background_scale : float or None or False, optional, default (1.)
        The scale factor of the background. If 'background' is None or 
        False then this is ignored.
        
    turn_off_bragg : bool, optional, default (False)
        If True then exclude the Bragg peaks from the diffraction volume
    
    turn_off_diffuse : bool, optional, default (False)
        If True then exclude the diffuse scatter from the diffraction volume
    
    Returns
    -------
    diff : numpy.ndarray, float, (unit_cell)
        The diffraction volume of the crystal. It is fftshifted so that the 
        zero frequency is at diff[0, 0, 0], as are all return arrays. To 
        place the zero frequency at the centre of the array (thus providing 
        a natural view of the sample) simply do numpy.fft.fftshift(diff).
    
    info : dictionary 
        info = {
           
        'beamstop' : numpy.ndarray, bool, (True = unmasked, False = masked)
            All True if no beamstop was defined
        
        'edges' : numpy.ndarray, bool, (True = unmasked, False = masked)
            All True if no edges were requested
        
        'support' : numpy.ndarray, bool, (True = contains solid unit, False = does not)
            may be padded if support_frac is not None or False
        
        'background' : numpy.ndarray, float 
            the background, all 0. if no background was requested
        
        'lattice' : numpy.ndarray, float 
            the lattice function of the crystal in the diffraction volume
        
        'voxels' : int
            the number of non-zero pixels in the solid_unit.
        
        'crystal' : numpy.ndarray, complex
            the real-space crystal function in the field-of-view
        
        'unit_cell' : numpy.ndarray, complex
            the real-space unit-cell function in the field-of-view
        
        'diffuse_weighting' : numpy.ndarray, float
            (1 - exp) 
        
        'Bragg_weighting' : numpy.ndarray, float
            N * exp * lattice
        
        'sym' : class object 
            an object for performing symmetry operations on the solid unit
    """
    # get the symmetry operator 
    ###########################
    if io_utils.isValid('space_group', params):
        space_group = params['space_group']
    else :
        space_group = 'P1'
    
    if space_group == 'P1' :
        sym_ops = symmetry_operations.P1(unit_cell, solid_unit.shape)
    
    elif space_group == 'P212121':
        sym_ops = symmetry_operations.P212121(unit_cell, solid_unit.shape)
    
    # define the solid_unit support
    ###############################
    if io_utils.isValid('support_frac', params):
        support = padding.expand_region_by(solid_unit > 0.1, params['support_frac'])
    else :
        support = np.abs(solid_unit) > 0.
    
    #solid_unit = np.random.random(support.shape)*support + 0J
    
    # propagate the solid unit to the detector
    ##########################################
    Solid_unit = np.fft.fftn(solid_unit)
    
    # generate the coppies of solid unit in the 
    # unit-cell
    ###########################################
    modes = sym_ops.solid_syms_Fourier(Solid_unit)
    
    # generate the diffuse and Bragg weighting modes
    ################################################
    exp     = make_exp(sigma, solid_unit.shape)
    lattice = symmetry_operations.lattice(unit_cell, solid_unit.shape)
    Bw      = N * exp * lattice 
    Dw      = (1. - exp) 
    
    # calculate the Bragg and diffuse scattering
    ############################################
    B      = Bw * np.abs(np.sum(modes, axis=0)**2)
    D      = Dw * np.sum(np.abs(modes)**2, axis=0)
    
    if io_utils.isValid('turn_off_bragg', params) :
        print('\nExluding Bragg peaks')
        B  = 0 
        Bw = 0
    elif io_utils.isValid('turn_off_diffuse', params) :
        print('\nExluding diffuse scattering')
        D  = 0
        Dw = 0
    
    diff = B + D
    
    # add photon counting noise
    ###########################
    if io_utils.isValid('photons', params) :
        diff, edges = add_noise_3d.add_noise_3d(diff, params['photons'], \
                                      remove_courners = params['cut_courners'],\
                                      unit_cell_size = unit_cell)
    else :
        edges = np.ones_like(diff, dtype=np.bool)
    
    # add gaus background
    #####################
    if io_utils.isValid('background', params) :
        sig   = params['background_std']
        scale = params['background_scale']
        scale *= (B+D).max()
        print('\nAdding gaussian to diff scale (absolute), std:', scale, sig)
        gaus  = gaus.gaus(solid_unit.shape, scale, sig)
        diff += gaus
        background = gaus
    else :
        background = np.zeros(solid_unit.shape, dtype=np.float)
     
    # add a beamstop
    ################
    if io_utils.isValid('beamstop', params) :
        print('making beamstop...')
        beamstop = beamstop.make_beamstop(solid_unit.shape, params['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)
        
    # testing
    #beamstop = lattice > 1.0e-1
    #diff    *= beamstop
    
    # make the unit-cell in the field-of-view
    #
    # make the crystal in the field-of-view
    #########################################
    crystal_ar, unit_cell_ar = sym_ops.solid_to_crystal_real(solid_unit, return_unit=True)
    
    voxels = np.sum(solid_unit > 0.)
    print('Simulation: number of voxels in solid unit:', voxels)
    print('Simulation: number of voxels in support   :', np.sum(support))
    
    info = {}
    info['beamstop']   = beamstop
    info['edges']      = edges
    info['support']    = support
    info['background'] = background
    info['lattice']    = lattice
    info['voxels']     = voxels
    info['crystal']    = crystal_ar
    info['unit_cell']  = unit_cell_ar
    info['sym']        = sym_ops
    
    return diff, info
