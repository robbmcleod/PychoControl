# -*- coding: utf-8 -*-
"""
For alignment of probe astigmatism by hand

Created on Wed Jul 17 16:20:07 2013

@author: Robert A. McLeod"""


import time
import TEM
import RAMutil as ram


""" For the most part using my TEM interface is easier here"""
tem = TEM.TEM_FeiTitan()
tem.connectToTEM()

while( True ):
    # Loop until I kill the thread
    probe = tem.acquireImage(tx =0.05, binning=[4,4], procmode=1)
    [probepos,probewidth] = ram.probeCenterAndAstig( probe )
    print( "Probe position : " + str(probepos) )
    print( "Probe width : " + str(probewidth) )
    print( "--------"  )
    time.sleep(0.05)
    
