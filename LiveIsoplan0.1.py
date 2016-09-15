# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:12:24 2014

@author: RM238934
"""


"""
Objective of this script is to take an image of the illumination (typically through
an amorphous region) AT LOWER MAGNIFICATION THAN YOU WANT TO OPERATE AT and then
do sub-area FFTs and plot them, live, to assess isoplanicity of the illumination.

Really should have a small pyQT dialogue?  
"""

import numpy as np
from PyQt4 import QtCore, QtGui
# import os
import sys
import TEM
import liUI
import functools
import DM3lib
import qimage2ndarray
import matplotlib.pyplot as plt

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s
    
class LiveIsoplan:
    
    def __init__(self):
        # VERBOSE declarations
        self.app = None
        self.MainWindow = None
        self.ui = None
        self.tem_interface = None
        self.tem_connected = False
        # self.reduce = 2 # Not currently implemented
        self.qpixbuffer = None
        self.running = False

        self.app = QtGui.QApplication(sys.argv)
        self.MainWindow = QtGui.QMainWindow()
        
        self.ui = liUI.Ui_IsoplanWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.ui.statusbar.showMessage( "Live Isoplanicity Measurement 0.1" )
        
        self.ui.actionFEI_Titan.triggered.connect( functools.partial( self.connectToTEM, 'FEITitan' ) )
        self.ui.actionSimulator.triggered.connect( functools.partial( self.connectToTEM, 'Simulator' )  )
        # self.ui.actionReInit.triggered.connect( self.reinitMicroscopeState )
        
        self.ui.pbLiveIso.clicked.connect( self.toggleLive )
        
        self.MainWindow.show() 
        sys.exit(self.app.exec_())
        # end __init__
        
    
    def toggleLive( self ):
        if self.tem_connected is False:
            self.ui.pbLiveIso.setChecked( False )
            self.ui.statusbar.showMessage( "Connect to TEM first..." )
            return
        """ Ok so if we clicked this we may need to start the loop, or interrupt it if it's already going.
        This is a little tricky because Python doesn't release control of the thread well...
        
        StackExchange may have ideas:
        http://doc.qt.digia.com/qq/qq27-responsive-guis.html
        
        """ 
        counter = 0
        while self.ui.pbLiveIso.isChecked():
            # Run Loop
            print "Running Live Isoplanicity loop iteration : " + str(counter)
            counter += 1
            
            mage = self.tem_interface.acquireImage( tx = self.ui.sbExposureTime.value(), binning = self.ui.sbBinning.value() )
            self.app.processEvents()
            self.updateIsoPlot( mage )
            self.app.processEvents()
        
        print "Ended Live Isoplanicity loop"
        
            
        

        
    def updateIsoPlot( self, mage ):
        # Do we pass in the data?  Presumably

        tile = np.array(mage.shape) / self.ui.sbNoTiles.value()
        tile2 = tile/2
        
        tilemage = np.zeros( np.array(mage.shape)/2 )
        for I in range(0,self.ui.sbNoTiles.value()):
            for J in range(0,self.ui.sbNoTiles.value()):
                fft_tile = np.fft.fft2(mage[I*tile[0]:(I+1)*tile[0],J*tile[1]:(J+1)*tile[1]])
                fft_tile[0,0] = 0;
                fft_tile = np.abs( np.fft.fftshift(fft_tile) )
        
                fft_tile = np.roll( np.roll( fft_tile, -fft_tile.shape[0]/4, axis=0), -fft_tile.shape[1]/4, axis=1 )
                fft_tile = fft_tile[0:fft_tile.shape[0]/2,0:fft_tile.shape[1]/2 ]
                
                tilemage[I*tile2[0]:(I+1)*tile2[0],J*tile2[1]:(J+1)*tile2[1]] = qimage2ndarray.numpy2uint8crop(fft_tile, cutoff = 0.1)
        
        # TO DO: is there a faster way to build a QPixmap rather than from an image first?
        self.qpixbuffer = QtGui.QPixmap.fromImage( qimage2ndarray.gray2qimage(tilemage, cutoff=0.0) )
        self.ui.labelPlot.setPixmap( self.qpixbuffer )
        
        
    def connectToTEM( self, temname ):
        # Connect to the FEITitan through the TNtem_feititan.py script
        if temname.lower() == 'feititan':
            self.tem_interface = TEM.TEM_FeiTitan()
            self.tem_connected = self.tem_interface.connectToTEM( self )
            if self.tem_connected is True:
                self.ui.statusbar.showMessage( "Successfully connected to FEI Titan" )
                # There doesn't seem to be an auto-exclude functionality as with radio buttons.
                self.ui.actionFEI_Titan.setChecked( True )
                self.ui.actionSimulator.setChecked( False )
            else:
                self.ui.statusbar.showMessage( "Failed to connect to FEI Titan" )
                self.ui.actionFEI_Titan.setChecked( False )
        if temname.lower() == 'simulator':
            self.tem_interface = TEM.TEM_Simulator()
            self.tem_connected = self.tem_interface.connectToTEM( self )
            self.ui.actionSimulator.setChecked( True )
            self.ui.actionFEI_Titan.setChecked( False )
            self.ui.statusbar.showMessage( "Connected to TEM simulator" )
    # end connectToTEM
            
liveisomain = LiveIsoplan()