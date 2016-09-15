# -*- coding: utf-8 -*-
"""
Created on Mon Aug 04 12:19:18 2014

@author: supervisor
"""


import numpy as np
import time
import TEM
import RAMutil as ram
from plotting import ims


""" For the most part using my TEM interface is easier here"""
tem = TEM.TEM_FeiTitan()
tem.connectToTEM()

# Ok output is uint16
# probe = tem.acquireImage( tx = 2.0, binning = 1, procmode=3 )


#profprobe_x = np.sum( probe, 0, dtype=float )
#profprobe_y = np.sum( probe, 1, dtype=float )
#
## Image must be background subtracted, or else we will blow up.  
#profprobe_x = ram.normalized(profprobe_x)
#profprobe_y = ram.normalized(profprobe_y)
#
#csprobe_x = np.cumsum(profprobe_x)
#csprobe_y = np.cumsum(profprobe_y)
#
#csprobe_x = ram.normalized(csprobe_x)
#csprobe_y = ram.normalized(csprobe_y)
#
#probepos_x = np.argwhere( csprobe_x >= 0.5 )[0][0]
#probepos_y = np.argwhere( csprobe_y >= 0.5 )[0][0]
#probepos = np.array( [probepos_y, probepos_x] )
#
#print( "Pos: " + str(probepos[0]) + ", " + str(probepos[1]) )
#
#upslope_y = np.nonzero( profprobe_y > 0.5)[0][0]
#downslope_y = np.nonzero( profprobe_y > 0.5)[0][-1]
#upslope_x = np.nonzero( profprobe_x > 0.5)[0][0]
#downslope_x = np.nonzero( profprobe_x > 0.5)[0][-1]
#
#probewidth = np.array( [downslope_y - upslope_y, downslope_x- upslope_x])
#
#print( "Width: " + str(probewidth[0]) + ", " + str(probewidth[1]) )