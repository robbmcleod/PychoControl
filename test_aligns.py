# -*- coding: utf-8 -*-
"""
Test script to understand how to build and access FEI's TEM Scripting COM object 
interface.

Created on Wed Jul 17 16:20:07 2013

@author: Robert A. McLeod"""

import numpy as np
#import h5py
from win32com.client import gencache
import win32com.client
import time
import matplotlib.pyplot as plt
import TEM
import RAMutil as ram
#import DM3lib
#import qimage2ndarray
#from PyQt4 import QtCore, QtGui
import pyDM


#temcomwrapper = gencache.EnsureModule('{BC0A2B03-10FF-11D3-AE00-00A024CBA50C}', 0, 1, 9)
titansim = gencache.EnsureDispatch('TEMScripting.Instrument') # Equivalent to connecting to the TEM
#titantia = win32com.client.Dispatch("ESVision.Application")

""" For the most part using my TEM interface is easier here"""
#tem = TEM.TEM_FeiTitan()
tem = TEM.TEM_FeiTitan()
tem.connectToTEM()

pyDM.getMessage()
pyDM.getMessage()
pyDM.getMessage()

foo = tem.acquireImage( procmode=3 )
plt.imshow(foo)
plt.show( block = False )

#intvect = np.linspace(-0.3,0.2,50)
#pyDM.sendMessage( "set_showimage_true" )
#
#tx = 1;
#m = 0;
#tem.setLens( 'int', intvect[m] )
#magename = "Probe-17k-10s-" + str(m)
#tem.acquireImage( name=magename, tx=tx, binning=[1,1] )



#while( True ):
#    # Loop until I kill the thread
#    probe = tem.acquireImage(tx =0.4, binning=[4,4], procmode=1)
#    [probepos,probewidth] = ram.probeCenterAndAstig( probe )
#    print( "Probe position : " + str(probepos) )
#    print( "Probe width : " + str(probewidth) )
#    print( "--------"  )
#    time.sleep(0.05)
    
# Call label.setPixmap( test )

#acqmages = [] # Clear the acquisition object
#acqmages.append(titansim.Acquisition.AcquireImages().Item(0))

"""
fhand = h5py.File( "test2.hdf5" ,'w') # file handle, saves in same directory as script
rotgrp = fhand.create_group( "rotation" )
rotgrp.attrs["foo"] = 5;
fhand.get('rotation').attrs.keys()
fhand.get('rotation').attrs.get('foo')
fhand.close()
"""


"""
# Let's test the xcorr2_fft that I wrote
magebase = np.double(tem.acquireImage( tx=2, binning=1, framesize='full', referencemode='default' ))

bs00 = tem.getBeamShift()
bsX0 = tem.getBeamShift()
bsX0[0] += 8E-9
tem.setBeamShift( bsX0 )

mageshift = np.double(tem.acquireImage( tx=2, binning=1, framesize='full', referencemode='default' ))

pos00 = bc.probeCenter( magebase )
posX0 = bc.probeCenter( mageshift )

# xc_shift = bc.xcorr2_fft( mageshift, magebase )

xc = np.abs( np.fft.ifftshift(np.fft.ifft2( np.fft.fft2(mageshift) * np.fft.fft2(magebase).conj()  )) )
ij = np.unravel_index(np.argmax(xc), xc.shape ) # Because Python is wierd
shift_xc = ij[::-1]
shift_xc -= np.divide( xc.shape, 2)
    
shift_com = posX0 - pos00

print "xc_shift = " + str(shift_xc) + ", com_shift = ", str(shift_com)
"""

# Let's try and see if we can get illumination to work
# temp = titansim.Illumination.Intensity
# print( str(temp ) )

# temp = 0.1;
# temp2 = titansim.Illumination.Intensity = temp;
# print( str(temp2) )

# This does in fact work, but what about dirrect assignment?  
# temp3 = titansim.Illumination.Intensity = 0.2
# print( str(temp3))
# It works!

# The objective defocus is in titansim.Projection.Defocus
# Values are in nominal nm of defocus

# What is titansim.Projection.Focus?  
# titansim.Projection.Focus is the diffraction focus.  By TEMspy it changes DL, 
# the diffraction lens (which is basically the first intermediate lens)
# Typical values are in the range of 0.1 to -0.1 ish
# Values are nominally nanometers but who knows really since it depends on convergence angle and camera length.