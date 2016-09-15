# -*- coding: utf-8 -*-
"""
Test script to understand how to build and access FEI's TEM Scripting COM object 
interface.

Created on Wed Jul 17 16:20:07 2013

@author: Robert A. McLeod"""

import numpy as np
#import h5py
#from win32com.client import gencache
#import win32com.client
import time
import matplotlib.pyplot as plt
import TEM
import RAMutil as ram
#import DM3lib
#import qimage2ndarray
#from PyQt4 import QtCore, QtGui
import pyDM


#temcomwrapper = gencache.EnsureModule('{BC0A2B03-10FF-11D3-AE00-00A024CBA50C}', 0, 1, 9)
#titansim = gencache.EnsureDispatch('TEMScripting.Instrument') # Equivalent to connecting to the TEM
#titantia = win32com.client.Dispatch("ESVision.Application")

""" For the most part using my TEM interface is easier here"""
#tem = TEM.TEM_FeiTitan()
tem = TEM.TEM_FeiTitan()
tem.connectToTEM()
intvect = np.linspace(-0.3,0.2,50)
pyDM.sendMessage( "set_showimage_true" )
pyDM.connect()
time.sleep(0.5)
diffvect = np.zeros( 50 )

m = 25
#tem.setLens( 'int', intvect[m] )
tem.setLens('diff', diffvect[m] )
pyDM.sendMessage( "set_showimage_false"  )




tx = 0.5
#diffvect[m] = tem.getLens('diff')
pyDM.sendMessage( "set_showimage_true" )
magename = "DiffSiCCA-580mm-0.5s-" + str(m)
testmage = tem.acquireImage( name=magename, tx=tx, binning=[1,1], proc=3 )


np.savetxt( 'intvect.txt', intvect )
np.savetxt( 'diffvect.txt', diffvect )