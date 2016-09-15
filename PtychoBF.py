# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:00:04 2014


@author: Robert A. McLeod

Ptychography Bright-field Scan 

This is a dialog box for PtychoControl

Takes a fast diffraction scan over an area, integrates the central peak of each diffraction pattern
to represent a pixel map of the bright-field intensity.   For alignment of precession electron diffraction
patterns and ptychography diffractive imaging scans.

Basically necessary because the FEI STEM system cannot be used at-will.

Currently trouble with the camera and garbage collection, appear to have a memory leak somewhere?
"""

# Basic question, it is worthwhile to have a seperate class here, or should I just have a giant one in PtychoControl?
class PtychoBF:
    
    
    def __init__( self ):
        pass
    