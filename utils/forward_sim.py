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
    if sigma is np.inf :
        print('sigma is inf setting exp = 0')
        exp     = np.zeros(i.shape, dtype=np.float)
    elif sigma is 0 :
        print('sigma is 0 setting exp = 1')
        exp     = np.ones(i.shape, dtype=np.float)
    else :
        try :
            exp = np.exp(-4. * np.pi**2 * ((sigma[0]*i)**2 +(sigma[1]*j)**2+(sigma[2]*k)**2))
        except TypeError: 
            exp = np.exp(-4. * sigma**2 * np.pi**2 * (i**2 + j**2 + k**2))
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

    lattice_blur : float, optional, default (None)
        If not 'None' then blur the lattice function with a gaussian of
        by 'lattice_blur' standard deviation (in pixels)
        
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
        
        'Bragg_weighting' : numpy.ndarray, float
            N * lattice * exp
            
        'diffuse_weighting' : numpy.ndarray, float
            (1 - exp)
        
        'solvent_content' : float
            The ratio: no. of voxels outside samples / number of voxels
        
        'solvent_content_support' : float
            The ratio: no. of voxels outside sample supports / number of voxels
        
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
    
    elif space_group == 'Ptest':
        sym_ops = symmetry_operations.Ptest(unit_cell, solid_unit.shape)
    
    # define the solid_unit support
    ###############################
    if io_utils.isValid('support', params):
        support = params['support']
    else :
        support = np.abs(solid_unit) > 0.
    
    if io_utils.isValid('support_frac', params):
        support = padding.expand_region_by(support, params['support_frac'])
    
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
    
    # make the lattice
    ##################
    lattice = symmetry_operations.make_lattice_subsample(unit_cell, solid_unit.shape, N)
    print('sum lattice:', np.sum(lattice))
    
    if io_utils.isValid('lattice_blur', params) :
        import scipy.ndimage
        lattice = scipy.ndimage.filters.gaussian_filter(lattice, params['lattice_blur'], truncate=10.)
        print('Bluring the lattice function...', lattice.dtype)
    
    # normalise the lattice by its max value
    #lattice = lattice / lattice.max()
    
    Bw      = exp * lattice 
    Dw      = (1. - exp) 
    
    # calculate the Bragg and diffuse scattering
    ############################################
    B      = Bw * np.abs(np.sum(modes, axis=0))**2
    D      = Dw * np.sum(np.abs(modes)**2, axis=0)
    
    if io_utils.isValid('turn_off_bragg', params) :
        print('\nExluding Bragg peaks')
        B.fill(0) 
        Bw.fill(0)
    if io_utils.isValid('turn_off_diffuse', params) :
        print('\nExluding diffuse scattering')
        D.fill(0)
        Dw.fill(0)
    
    Bsum = np.sum(B)
    Dsum = np.sum(D)
    print('integrated intensity of the Bragg reflections:', Bsum)
    print('integrated intensity of the diffuse scatter  :', Dsum)
    print('fractional total intensity of the Bragg reflections:', Bsum / (Bsum+Dsum))
    print('fractional total intensity of the diffuse scatter  :', Dsum / (Bsum+Dsum))
    
    # add photon counting noise
    ###########################
    if io_utils.isValid('photons', params) :
        # R-scale data
        if not io_utils.isValid('turn_off_bragg', params) :
            B_rscale, R_scale = add_noise_3d.R_scale_data(B)
            B_norm    = np.sum(B_rscale)
        else :
            B_norm    = 0
        
        if not io_utils.isValid('turn_off_diffuse', params) :
            D_rscale, R_scale = add_noise_3d.R_scale_data(D)
            D_norm    = np.sum(D_rscale)
        else :
            D_norm    = 0
        
        # add noise
        norm      = B_norm + D_norm

        # R-scale data
        if not io_utils.isValid('turn_off_bragg', params) :
            B_photons = params['photons'] * B_norm / norm
            B_rscale = add_noise_3d.add_poisson_noise(B_rscale, B_photons)
            # un-scale 
            B_rscale /= R_scale
            # renormalse
            B = B_rscale / np.sum(B_rscale) * Bsum
        else :
            B_photons = 0
        
        if not io_utils.isValid('turn_off_diffuse', params) :
            D_photons = params['photons'] * D_norm / norm
            D_rscale = add_noise_3d.add_poisson_noise(D_rscale, D_photons)
            # un-scale 
            D_rscale /= R_scale
            # renormalse
            D = D_rscale / np.sum(D_rscale) * Dsum
        else :
            D_photons = 0
        
        print('\nnumber of photons for Bragg   diffraction:', B_photons)
        print('number of photons for diffuse diffraction:', D_photons)
        print('total number of photons for diffraction  :', params['photons'])
        assert( np.abs((B_photons + D_photons) - params['photons']) < 1)
        
        diff = B + D
    else :
        diff = B + D

    if io_utils.isValid('cut_courners', params) :
        edges = add_noise_3d.mask_courners(diff.shape)
        diff *= edges
        B    *= edges
        D    *= edges
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
        
    # make the unit-cell in the field-of-view
    # make the crystal in the field-of-view
    #########################################
    crystal_ar, unit_cell_ar = sym_ops.solid_to_crystal_real(solid_unit, return_unit=True)
    
    # no. of voxels and solvent content
    voxels           = np.sum(np.abs(solid_unit) > 0.)
    voxels_sup       = np.sum(support > 0.)
    voxels_unit_cell = np.prod(unit_cell)
    solvent_sample   = 1. - sym_ops.no_solid_units * voxels     / float(voxels_unit_cell)
    solvent_support  = 1. - sym_ops.no_solid_units * voxels_sup / float(voxels_unit_cell)
    
    print('Simulation: number of voxels in solid unit:', voxels)
    print('Simulation: number of voxels in support   :', voxels_sup)
    print('Simulation: solvent fraction (sample)     :', solvent_sample)
    print('Simulation: solvent fraction (support)    :', solvent_support)
    
    U = np.fft.ifftn(np.sum(modes, axis=0))

    sU = np.sum(np.abs(U)>1.0e-5)
    sS = voxels
    if sU < sym_ops.no_solid_units * sS :
        print('##########################')
        print('Warning: crystal overlap!!')
        print('of', sym_ops.no_solid_units * sS - sU, 'voxels')
        print('##########################')

    info = {}
    info['beamstop']   = beamstop
    info['edges']      = edges
    info['support']    = support
    info['background'] = background
    info['lattice']    = lattice
    info['voxels']     = voxels
    info['crystal']    = crystal_ar
    info['unit_cell']  = U
    info['sym']        = sym_ops
    info['modes']      = modes
    info['Bragg_weighting']   = Bw
    info['diffuse_weighting'] = Dw
    info['Bragg_diffraction']   = B
    info['diffuse_diffraction'] = D
    info['solvent_content'] = solvent_sample
    info['solvent_content_support'] = solvent_support
    
    return diff, info
