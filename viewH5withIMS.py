# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 17:39:03 2014

@author: supervisor
"""

import tables
from plotting import ims

# filename = "test_IBSimage.hdf5"
#filename = "test_IBSimage.hdf5"
# filename = "AuOnC_probeA.hdf5"
filename = "AuOnC_probeC_1.5nmstep.hdf5" 

h5file = tables.open_file( filename, mode='r' )

#imagematrix = h5file.get_node( "/imageshiftsA_diff"  )

imagematrix = h5file.get_node( "/probeA_diff" )

ims( imagematrix, cutoff=0.001 )