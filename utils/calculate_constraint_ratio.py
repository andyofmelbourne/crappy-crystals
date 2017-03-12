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

# calculate the constraint ratio in phase retrieval based 
# on the crystal space group and the solid-unit support
# assuming an oversampled solid-unit 
# and assuming that the unit-cell fits within the field-of-view

# constraint ratio = volume of symmetry summed autocorrelation + volume of cross-correlation terms in the unit-cell volume
#                  / number of symmetries in the Patterson space group * volume of solid-unit

def calculate_constraint_ratio(support, space_group, unit_cell_size):
    """
    Assume that support is defined in the field-of-view
    """
    # make the symmetry group operator
    if space_group == 'P1' :
        sym_ops = symmetry_operations.P1(unit_cell_size, support.shape)
    
    elif space_group == 'P212121':
        sym_ops = symmetry_operations.P212121(unit_cell_size, support.shape)
    
    # probably don't need double prec.
    o = support.astype(np.complex64) 
    O = np.fft.fftn(o)

    # calculate symmetry related coppies of o
    Us = sym_ops.solid_syms_Fourier(O)
    
    # calculate the symmetry summed autocorelation
    A_o = np.sum((Us.conj() * Us).real, axis=0)
    A_o = np.fft.ifftn(A_o)
    
    # calculate the symmetry summed autocorelation support
    A_sup = A_o.real > 0.5
    
    # calculate the aliased cross-correlation 
    #u_sup   = np.sum(sym_ops.solid_syms_real(support), axis = 0 ) > 0.5
    u_sup   = np.zeros_like(support)
    u_sup[:unit_cell_size[0], :unit_cell_size[1], :unit_cell_size[2]] = True
    u_sup = np.fft.ifftshift(u_sup)
    lattice = symmetry_operations.lattice(unit_cell_size, support.shape)
    U   = np.sum(Us, axis=0)
    Pat = (U.conj() * U).real * lattice
    C   = (Pat - np.sum((Us.conj() * Us).real, axis=0)) * lattice
    C   = np.fft.ifftn(C)
    Pat = np.fft.ifftn(Pat)
    
    # calculate the aliased cross-correlation support
    C_sup   = C.real[:unit_cell_size[0], :unit_cell_size[1], :unit_cell_size[2]] > 0.5
    Pat_sup = Pat.real[:unit_cell_size[0], :unit_cell_size[1], :unit_cell_size[2]] > 0.5

    # calculate constraint ratio
    den       = float(sym_ops.Pat_sym_ops * np.sum(support))
    num_con   = float(np.sum(A_sup))
    num_C     = np.sum(C_sup)
    num_Pat   = np.sum(Pat_sup)
    omega_con    = num_con / den   
    omega_Bragg  = num_Pat / den 
    omega_C      = num_C / den 
    omega_global = (num_con + num_C) / den
    
    return omega_con, omega_Bragg, omega_global
