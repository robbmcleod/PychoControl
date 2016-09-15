# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:14:11 2014

PtychoControl interface.  An interface for using a TEM (nominallly an FEI Titan)
for both acquisition and calibration for diffractive imaging.

@author: Robert A. McLeod
"""

# import matplotlib
# TO DO: install pygtk
#matplotlib.use('GTKAgg')
import numpy as np
from PyQt4 import QtCore, QtGui
# import os
import sys
import TEM
import pcUI
import ptychoBF_UI
import RAMutil as ram
import matplotlib.pyplot as plt
import h5py
# from skimage.feature import match_template
import time
import functools
import qimage2ndarray as q2np

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s
    
class PtychoControl:


    def __init__( self ):
        self.app = QtGui.QApplication(sys.argv)
        self.MainWindow = QtGui.QMainWindow()
        self.BrightDialog = QtGui.QDialog()

        self.ui = pcUI.Ui_pcMainWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.tem = None
        self.tem_connected = False
        
        self.descan_probe_temp = None
        self.descan_probe_saveable = None
        self.descan_diff_temp = None
        self.diffshift_focus = None
        self.diffshift_aperture = None
        self.descan_diff_saveable = None
        self.manual_diff_init = None
        self.last_set_focus = False
        
        self.position_vacuum = None
        self.position_crystal = None
        self.DL_manual = None
        self.int_counter = 0    
        self.dset_vacuumfocus = None
        self.dset_vacuumaperutre = None
        self.dset_crystalfocus = None
        self.dset_crystalaperture = None
        
        # Rotation and descan calibration
        self.rot_image = None
        self.rot_diff = None
        self.rot_descan = None
        self.rot_gamma = None # Rotation between image space and diffraction space
        
        # The dose from the 
        self.doselevel = 0
        self.N = 2048 # detector pixel length
        
        self.defocus_vector = None
        self.intensity_vector = None
        
        self.hdf5handle = []
        
        self.ptycho_bs_pos = None # The associated beam shifts with the Ptychography scan
        self.ptycho_ds_pos = None # The associated descans
        
        self.savefilename = ""
        
        self.ui.statusbar.showMessage( "Welcome to PtychoControl 0.1!" )
        
        self.ui_ptychoBF = ptychoBF_UI.Ui_PtychoBFDialog()
        self.ui_ptychoBF.setupUi( self.BrightDialog )
        
        # Apply all the connections from the QT gui objects to their associated functions in this class
        self.ui.actionFEI_Titan.triggered.connect( functools.partial( self.connectToTEM, 'FEITitan' ) )
        self.ui.actionSimulator.triggered.connect( functools.partial( self.connectToTEM, 'Simulator' )  )
        
        # Calibration tab
        QtCore.QObject.connect(self.ui.pbDescanBS, QtCore.SIGNAL("clicked()"), self.calibrateDescan )
        QtCore.QObject.connect(self.ui.pbDescanDS, QtCore.SIGNAL("clicked()"), self.calibrateDescanDS )
        QtCore.QObject.connect(self.ui.pbAcquireSeries, QtCore.SIGNAL("clicked()"), self.acquireCaliSeries )
        QtCore.QObject.connect(self.ui.pbOpenFile, QtCore.SIGNAL("clicked()"), self.setFileName )
        QtCore.QObject.connect(self.ui.rbSaveOffsetsProbe, QtCore.SIGNAL("clicked()"), self.storeProbeOffsets )
        QtCore.QObject.connect(self.ui.rbSaveOffsetsDiff, QtCore.SIGNAL("clicked()"), self.storeDiffOffsets )
        QtCore.QObject.connect(self.ui.pbTestGotoStart, QtCore.SIGNAL("clicked()"), self.gotoStart)
        QtCore.QObject.connect(self.ui.pbTestGotoEnd, QtCore.SIGNAL("clicked()"), self.gotoEnd)
        
        # Manual-assisted diffraction series
        QtCore.QObject.connect(self.ui.pbSetDLFocus, QtCore.SIGNAL("clicked()"), self.setDLFocus)        
        QtCore.QObject.connect(self.ui.pbSetDLAperture, QtCore.SIGNAL("clicked()"), self.setDLAperture) 
        QtCore.QObject.connect(self.ui.pbGotoDLFocus, QtCore.SIGNAL("clicked()"), self.gotoDLFocus)        
        QtCore.QObject.connect(self.ui.pbGotoDLAperture, QtCore.SIGNAL("clicked()"), self.gotoDLAperture)
        QtCore.QObject.connect(self.ui.pbSetVacuumArea, QtCore.SIGNAL("clicked()"), self.setVacuumArea) 
        QtCore.QObject.connect(self.ui.pbSetCrystalArea, QtCore.SIGNAL("clicked()"), self.setCrystalArea) 
        QtCore.QObject.connect(self.ui.pbGotoVacuumArea, QtCore.SIGNAL("clicked()"), self.gotoVacuumArea) 
        QtCore.QObject.connect(self.ui.pbGotoCrystalArea, QtCore.SIGNAL("clicked()"), self.gotoCrystalArea) 
        QtCore.QObject.connect(self.ui.pbAcquireVacuumDiff, QtCore.SIGNAL("clicked()"), self.acquireVacuumDiff) 
        QtCore.QObject.connect(self.ui.pbAcquireCrystalDiff, QtCore.SIGNAL("clicked()"), self.acquireCrystalDiff)
        QtCore.QObject.connect(self.ui.pbNextIntensity, QtCore.SIGNAL("clicked()"), self.nextIntensity) 
        QtCore.QObject.connect(self.ui.pbPrevIntensity, QtCore.SIGNAL("clicked()"), self.prevIntensity) 
        QtCore.QObject.connect(self.ui.pbManualDiff, QtCore.SIGNAL("clicked()"), self.startDiffManual) 
        
        # Control tab
        QtCore.QObject.connect(self.ui.pbPtychoCalcPos, QtCore.SIGNAL("clicked()"), self.ptychoGenerateCoords)
        QtCore.QObject.connect(self.ui.pbAcquirePtycho, QtCore.SIGNAL("clicked()"), self.acquirePtycho )
        
        # Image/diffraction Rotation tab
        QtCore.QObject.connect(self.ui.pbRotUpdate, QtCore.SIGNAL("clicked()"), self.rotationUpdate)
        QtCore.QObject.connect(self.ui.pbRotCalibrate, QtCore.SIGNAL("clicked()"), self.rotationCalibration)
        QtCore.QObject.connect(self.ui.pbRotTestDescan, QtCore.SIGNAL("clicked()"), self.rotationTestDescan)
        self.ui.pbRotLoad.clicked.connect( self.rotationLoad )
        self.ui.pbRotSave.clicked.connect( self.rotationSave )
        
        # RAMdisk
        self.ui.leTempFileLocation.textChanged.connect( self.setTempFileLoc  )
        self.ui.leTempFileLocation.setEnabled( False )
        
        self.ui.pbZeroBSDS.clicked.connect( self.zeroBSandDS )
        self.ui.pbPrintAll.clicked.connect( self.printAll )
        
        # Bright-field dialog
        self.ui.pbBFScan.clicked.connect( self.BFScan )
        self.ui.pbBFCancel.clicked.connect( self.BFCancel )
        self.ui_ptychoBF.tbCrosshair.clicked.connect( functools.partial( self.BFInitCrosshair ) )
        self.ui_ptychoBF.tbGoto.clicked.connect( functools.partial( self.BFGoTo ) )
        self.ui_ptychoBF.tbRefresh.clicked.connect( functools.partial( self.BFRefresh ) )
        self.BF_sentinel = True
        
        # The stored brightfield image
        self.bfmatrix = np.zeros( [ self.ui.sbXScanCount.value(), self.ui.sbYScanCount.value() ])
        
        
        self.MainWindow.show() 
        self.BrightDialog.show()
        sys.exit(self.app.exec_())
        
    # end __init__
        
    def printAll( self ):
        print "rot_image : " + str(self.rot_image)
        print "rot_descan (of beamshift) : " + str(self.rot_descan)
        print "rot_diff : " + str(self.rot_diff)
        print "rot_gamma : " + str(self.rot_gamma)
        print "doselevel : " + str(self.doselevel)
        
        
    def zeroBSandDS( self ):
        self.tem.setDeflector( "ds",  np.array([0.0,0.0]) )
        self.tem.setDeflector( "bs",  np.array([0.0,0.0])  )
        
    def setTempFileLoc( self ):
        print "Setting temporary file location to : " + str(self.ui.leTempFileLocation.text())
        self.tem.setTempLocation( str(self.ui.leTempFileLocation.text()) )
        
        """
    def closeEvent( self ):
        print "Closing Ptyho Control..."
        self.DisplayDialog.deleteLater()
        self.MainWindow.deleteLater()
        self.app.deleteLater()
        """
        
    def rotationUpdate( self ):
        print "DEBUG: rotationUpdate()" 
        if self.tem_connected is False:
            self.ui.statusbar.showMessage( "Connect to microscope first." )
            return

        # Note that imagmag and difflen are bullshit values in the current implementation of TEMScripting, so 
        # they cannot be used.  We would have to load the nominal values externally.
        [imageindex, imagemag] = self.tem.getImageMag()
        self.ui.sbRotMagImage.setValue( imageindex )
        
        [diffindex, difflen] = self.tem.getDiffractionLength()
        self.ui.sbRotMagDiff.setValue( diffindex )
    #end rotationUpdate
        
    def rotationCalibration( self ):
        if self.tem_connected is False:
            self.ui.statusbar.showMessage( "Connect to microscope first." )
            return
        
        self.tem.setMode('image')
        [imageindex, imagemag] = self.tem.getImageMag()
        self.ui.sbRotMagImage.setValue( imageindex )
        
        # Do a calibrate descan
        bs_imageshift=self.ui.sbRotBSImage.value()*1E-9 # convert from nm to m
        _, self.rot_image = self.calibrateDescan( deflector="bs", def_shift=bs_imageshift, doCentering=True ) 
        # rot_image[0,:] is image shift (pix) for BSX
        # rot_image[1,:] is image shift (pix) for BSY
        
        self.tem.setMode('diffraction')
        [diffindex, difflen] = self.tem.getDiffractionLength()
        self.ui.sbRotMagDiff.setValue( diffindex )
        
        # Now in diffraction mode let's see what the relationship is between the beamshifts and 
        # diffraction shifts
        ds_descan = self.ui.sbRotDSDescan.value() * 1E-3 # convert from mm to m
        # Settling time
        _, self.rot_descan = self.calibrateDescan( deflector="ds", def_shift=ds_descan, doCentering=True )
        
        # Now find out how much the beam shifts move the beam at the diffraction plane (mostly tilt in fact)
        curr_bs = self.tem.getDeflector( "bs" )
        self.app.processEvents() # Unfreeze the GUI
        
        plt.pause( 2.0 ) # Much better function than time.sleep()
        bs_diffshift = self.ui.sbRotBSDiff.value()*1E-9 # convert from nm to m
        _, self.rot_diff = self.calibrateDescan( deflector="bs", def_shift=bs_diffshift, doCentering=False )
        self.tem.setDeflector( "bs", curr_bs ) # Reset beamshift      
        
        # Don't bother to change back?  Doesn't fit the workflow
        # self.tem.setMode( initialmode )
        
        # See descanBSwithDS() for information on how to descan
        
        # Find the rotational vectors in x and y
        theta_ix = np.arctan2( self.rot_image[0,1], self.rot_image[0,0] )
        theta_dx = np.arctan2( self.rot_diff[0,1], self.rot_diff[0,0] )
        theta_iy = np.arctan2( self.rot_image[1,1], self.rot_image[1,0] )
        theta_dy = np.arctan2( self.rot_diff[1,1], self.rot_diff[1,0] )
        # Find deltas
        self.rot_gamma = np.mean( [np.mod( theta_dx - theta_ix, np.pi ), np.mod( theta_dy - theta_iy, np.pi )] )
    
        print "Angle between image- and diffraction-plane = " + str( 180/np.pi * self.rot_gamma ) + " degrees" 

        self.ui.labelRotCali.setText( 'CALIBRATED' )
    #end rotationCalibration
        
    def rotationTestDescan( self ):
        # Should be in diffracton mode
        self.tem.setMode( 'diffraction' )
        init_bs = self.tem.getDeflector( "bs" )
        init_ds = self.tem.getDeflector( "ds" )
        
        bintest = self.ui.sbTestBinning.value()
        
        # Get an diffractogram
        tx_test = self.ui.sbTargetCountsDiff.value() / self.doselevel / bintest**2
        
        mage1 = self.tem.acquireImage( tx_test, bintest )  
        base_pos = ram.probeCenter( mage1 )
        
        # Apply a big beam shift
        big_bs = np.array( [self.ui.sbRotBSDiff.value()*1E-9, -0.5*self.ui.sbRotBSDiff.value()*1E-9 ] )
        self.tem.setDeflector( "bs",  big_bs + init_bs )
        mage2 = self.tem.acquireImage( tx_test, bintest )  
        shift_pos = ram.probeCenter(mage2)
        
        self.descanBSwithDS( big_bs )
        
        # Get a descanned diffractogram
        mage3 = self.tem.acquireImage( tx_test, bintest )  
        descan_pos = ram.probeCenter(mage3)
        
        xc_shift = ram.xcorr2_fft( mage3, mage1 )
        
        # Reset microscope state
        self.tem.setDeflector( "bs", init_bs )
        self.tem.setDeflector( "ds", init_ds )
        print 'Beam shift of : ' + str( big_bs )
        print 'base: ' + str(base_pos) + ', shifted: ' + str(shift_pos) + ', descanned: ' + str(descan_pos)
        
        print 'COM shift was: ' + str( shift_pos - base_pos )
        print 'descan was: ' + str( descan_pos - shift_pos )
        print 'COM shift error was : ' + str( descan_pos - base_pos )
        print 'XC shift error was: ' + str( xc_shift )
        
        # plt.imshow( mage1 + mage2 )
        fig, (ax1, ax2, ax3) = plt.subplots( 1, 3, sharey=True )
        ax1.imshow( mage1 )
        pylab.axis( 'equal' )
        ax2.imshow( mage2 )
        pylab.axis( 'equal' )
        pylab.title( 'base: ' + str(base_pos) + ', shifted: ' + str(shift_pos) + ', descanned: ' + str(descan_pos) )
        ax3.imshow( mage3 )
        pylab.axis( 'equal' )
        
        plt.show()
        
    #end rotTestDescan
        
    def descanBSwithDS( self, input_bs ):
        """ 
        BASED ON TESTING THE DESCAN IS ONLY ACCURATE TO WITHIN ABOUT 40 PIXELS
        I would like to know why it isn't perfect...
        """
        print "FIXME: doing descan relative to current DS is dangerous"  
        init_ds = self.tem.getDeflector( "ds" )
        
        delta_pix = ( self.rot_diff[0,:] * -input_bs[0] ) + ( self.rot_diff[1,:] * -input_bs[1] )
        print 'DescanBSwithDS attempting to remove: ' + str(delta_pix)      
        
        # Then figure out how much ds_descan has to be applied to remove it.
        # kfact = (ds_c[1,1]*delta[0] - ds_c[1,0]*delta[1]) / (-ds_c[0,1]*ds_c[1,0] + ds_c[0,0]*ds_c[1,1] )
        # lfact = (-ds_c[0,1]*delta[0] + ds_c[0,0]*delta[1]) / (-ds_c[0,1]*ds_c[1,0] + ds_c[0,0]*ds_c[1,1] )
        kfact = (self.rot_descan[1,1]*delta_pix[0] - self.rot_descan[1,0]*delta_pix[1]) / (-self.rot_descan[0,1]*self.rot_descan[1,0] + self.rot_descan[0,0]*self.rot_descan[1,1] )
        lfact = (-self.rot_descan[0,1]*delta_pix[0] + self.rot_descan[0,0]*delta_pix[1]) / (-self.rot_descan[0,1]*self.rot_descan[1,0] + self.rot_descan[0,0]*self.rot_descan[1,1] )
        print 'Applying DS of ' + str(kfact) + 'X and ' + str(lfact) + 'Y'  
        
        descan_nominal = np.array( [init_ds[0] + kfact, init_ds[1] + lfact] )
        
        self.tem.setDeflector( "ds", descan_nominal )
        
    def rotationSave( self ):
        rotfilestr = str(QtGui.QFileDialog.getSaveFileName(self.MainWindow,"Save Rotation Calibration", "", "HDF5 Files (*.hdf5)"))
        rotfh = h5py.File( rotfilestr, 'a')
        if bool( rotfh  ):
            # Try to create a rotation group
            if not bool( rotfh.get( 'rotation' ) ):
                rotgrp = rotfh.create_group( 'rotation' )
            else:
                rotgrp = rotfh.get( 'rotation' ) # yes, I know I do not have to call this twice
            
            rotgrp.attrs['magnification'] = self.ui.sbRotMagImage.value()
            rotgrp.attrs['difflen_index'] = self.ui.sbRotMagDiff.value()
            rotgrp.attrs['intensity'] = self.tem.getLens( "int" )
            rotgrp.attrs['diffractionlens'] = self.tem.getLens( "diff" )
            rotgrp.attrs['rot_image'] = self.rot_image
            rotgrp.attrs['rot_diff'] = self.rot_diff
            rotgrp.attrs['rot_descan'] = self.rot_descan
            rotgrp.attrs['rot_gamma'] = self.rot_gamma
            rotgrp.attrs['doselevel'] = self.doselevel
            rotfh.close()
        else:
            self.ui.statusbar.showMessage( "WARNING: Rotation calibration not saved to HDF5 file" )
            
    def rotationLoad( self ):
        rotfilestr = str(QtGui.QFileDialog.getOpenFileName(self.MainWindow,"Load Rotation Calibration", "", "HDF5 Files (*.hdf5)"))
        rotfh = h5py.File( rotfilestr, 'r')
        if bool( rotfh ):
            if not bool( rotfh.get( 'rotation' ) ):
                self.ui.statusbar.showMessage( "WARNING: cannot load rotation calibrations, not found!" )
                rotfh.close()
                return
            else:
                rotgrp = rotfh.get( 'rotation' ) # yes, I know I do not have to call this twice
                
            self.rot_image = rotgrp.attrs.get('rot_image')
            self.rot_diff = rotgrp.attrs.get('rot_diff')
            self.rot_descan = rotgrp.attrs.get('rot_descan')
            self.rot_gamma = rotgrp.attrs.get('rot_gamma')
            self.doselevel = rotgrp.attrs.get('doselevel')
            rotfh.close()
            self.ui.labelRotCali.setText( "CALIBRATED" )
        else:
            self.ui.statusbar.showMessage( "WARNING: Rotation calibration not loaded from HDF5 file" )
             
        
    def meterBeamIntensity( self ):
        # Find and set doselevel
        print "TO DO: set-up independant script for light-metering the beam intensity (quickly)"
        
        
    def calibrateDescan( self, deflector = "bs", def_shift = [], doCentering = True ):
        # Take three test images to determine appropriate exposure levels and descan
        # Ok now to start we want to calibrate the system  with a couple of test shifts 
        # and use that to calibrate to centre the probe for the rest of the series
        #
        # Deflector can be:
        #   ds = diffraction shift
        #   bs = beam shift
        #
        # WE MUST TAKE THREE PICTURES BECAUSE DS-X AND DS-Y ARE NOT ORTHONORMAL
        # (and even BS-X and BS-Y are not exactly so)
        
        if self.tem_connected == False:
            self.ui.statusbar.showMessage( "Error: not connected to TEM" )
            return
    
        tx_test = self.ui.sbTestTx.value()
        binning_test = self.ui.sbTestBinning.value()
        
        # def_shift can be passed in, or if not get from the ui elements
        if not bool( def_shift ):
            if( deflector is "ds" ):
                def_shift = self.ui.sbTestDiffShift.value()
            elif( deflector is "bs" ):
                def_shift = self.ui.sbTestBeamShift.value() * 1E-9
        
        print "Deflector test shift = " + str(def_shift) 
        
        init_shift = self.tem.getDeflector( deflector )
        
        # Take the reference position
        self.ui.statusbar.showMessage( "calibrateDescan: taking reference position" )
        # print "Image 1 : " + str(init_shift)
        testprobe00 = self.tem.acquireImage( tx_test, binning_test )
        self.app.processEvents() # Unfreeze the GUI
        
        # plotmage = plt.imshow(testprobe00)
        # plt.show()
        
        # Shift in X
        self.ui.statusbar.showMessage( "calibrationDescan: test x-axis shift" )
        test_shift = np.array([init_shift[0]+def_shift,init_shift[1]]) 
        self.tem.setDeflector( deflector, test_shift)
        
        # print "Image 2 : " + str(test_shift)
        testprobeX0 = self.tem.acquireImage( tx_test, binning_test )
        self.app.processEvents() # Unfreeze the GUI
        
        # Shift in Y
        self.ui.statusbar.showMessage( "calibrationDescan: test y-axis shift" )
        test_shift = np.array([init_shift[0],init_shift[1] + def_shift ]) 
        self.tem.setDeflector( deflector, test_shift)
           
        # print "Image 3 : " + str(test_shift)
        testprobe0Y = self.tem.acquireImage( tx_test, binning_test )
        self.app.processEvents() # Unfreeze the GUI
        
        # find the centers
        # Could also use CoM from PIL or one of the Hyperspy methods here...
        # Or template-matching from scikit-image
       
        # Seems like probeCenter is quite effective...
        pos00 = ram.probeCenter( testprobe00 )
        posX0 = ram.probeCenter( testprobeX0 )
        pos0Y = ram.probeCenter( testprobe0Y )
        # print "00 : " + str(pos00) +", X0 : " + str(posX0) + ", 0Y : " + str(pos0Y)
        
        # xc_xshift = ram.xcorr2_fft( testprobeX0, testprobe00 )
        # xc_yshift = ram.xcorr2_fft( testprobe0Y, testprobe00 )
        
        # print "delta_xc(x) = " + str(xc_xshift)
        # print "delta_xc(y) = " + str(xc_yshift)
        print "delta_com(x) = " + str(pos00 - posX0)
        print "delta_com(y) = " + str(pos00 - pos0Y)
        
        # This is a lazy way to use class variables, but whatever...
        self.doselevel = 0.3333*( np.double(np.max(testprobe00)) + np.double(np.max(testprobeX0)) + np.double(np.max(testprobe0Y))  )
        # Normalize to 1 second exposure equivalent
        self.doselevel = self.doselevel / tx_test / (binning_test/self.ui.sbBinning.value() )**2
        print "Dose level = " + str(self.doselevel)
        
        # Now set the aligns so the first picture will be at (0,0)
        ds_c = np.zeros([2,2])
        # CENTER OF MASS
        ds_c[0,:] = binning_test*(pos00 - posX0) / def_shift # gives shift in x and y in pixel per nominal FEI unit, for beam shift X
        ds_c[1,:] = binning_test*(pos00 - pos0Y) / def_shift # gives shift in x and y in pixel per nominal FEI unit, for beam shift Y
        # XCORR2_FFT
        # ds_c[0,:] = binning_test*(xc_xshift) / def_shift # gives shift in x and y in pixel per nominal FEI unit, for beam shift X
        # ds_c[1,:] = binning_test*(xc_yshift) / def_shift # gives shift in x and y in pixel per nominal FEI unit, for beam shift Y
        
        # print "ds_c : " + str(ds_c)
        # print "ds_mt : " + std(ds_mt)
        
        # Push the probe back to the center of the screen
        # CENTER OF MASS CODE
        
        delta = np.double(pos00) - np.divide(testprobe00.shape, 2.0)
        # HEREIN LIES A BUG: ds_c is in binning 1 pixels...
        delta = delta * binning_test
        
        # print "delta : " + str(delta)
        kfact = (ds_c[1,1]*delta[0] - ds_c[1,0]*delta[1]) / (-ds_c[0,1]*ds_c[1,0] + ds_c[0,0]*ds_c[1,1] )
        lfact = (-ds_c[0,1]*delta[0] + ds_c[0,0]*delta[1]) / (-ds_c[0,1]*ds_c[1,0] + ds_c[0,0]*ds_c[1,1] )
        # print "kfact : " + str(kfact) + ", lfact : " + str(lfact)
        delta_nominal = np.array( [init_shift[0] + kfact, init_shift[1] + lfact] )
        # print "delta_nominal : " + str(delta_nominal)
        if doCentering:
            self.tem.setDeflector( deflector, delta_nominal )

        # save the descan parameter, by returning it
        # REMEMBER THAT THE DETECTOR IS BINNED.  For this application (centering) it doesn't matter, but
        # if you needed to shift a given number of pixels it would be.
        return delta_nominal, ds_c

    #end calibrateDescan

    def startDiffManual( self ):
        # Need four matrices in HDF5: vacfocus, vacaperture, crystalfocus, crystalaperture
        # A counter to move up and down intensity_vector
        # Stage positions for vacuum and crystal
        if self.tem_connected is False:
            self.ui.statusbar.showMessage( "Connect to microscope before starting manual-assisted diffraction series" )
            return
        
        self.manual_diff_init = True
        self.int_counter = 0;
        
        intensity_start = self.ui.sbIntensityStart.value()
        intensity_end = self.ui.sbIntensityEnd.value()
        msteps = self.ui.sbIntensitySteps.value()
        self.intensity_vector = np.linspace( intensity_start, intensity_end, msteps )
        self.DL_manual = np.zeros( [np.size(self.intensity_vector), 2]) # focus , aperture
        self.diffshift_aperture = 999*np.ones( [np.size(self.intensity_vector), 2])
        self.diffshift_focus = 999*np.ones( [np.size(self.intensity_vector), 2])
        
        # TO DO: update intensity and label in one step
        self.tem.setLens( "int", self.intensity_vector[self.int_counter] )
        self.setIntensityLabel( self.tem.getIntensity() )
        
        self.hdf5handle = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        # File compression is quite slow unfortunately
        # probe_dset = fhand.create_dataset( "probe_data", (self.N, self.N, msteps), chunks=(self.N, self.N, 1), compression="gzip" )
        print "TO DO: add error checking for hdf5 dataset generation"
        self.dset_vacuumfocus = self.hdf5handle.create_dataset( "vacuum_focus", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        self.dset_vacuumaperture = self.hdf5handle.create_dataset( "vacuum_aperture", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        self.dset_crystalfocus = self.hdf5handle.create_dataset( "crystal_focus", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        self.dset_crystalaperture = self.hdf5handle.create_dataset( "crystal_aperture", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        
        # Swap button functionality
        self.ui.pbManualDiff.setText( "Manual Diff END" )
        self.ui.pbManualDiff.setChecked( True )
        self.app.processEvents()
        QtCore.QObject.disconnect(self.ui.pbManualDiff, QtCore.SIGNAL("clicked()"), self.startDiffManual)
        QtCore.QObject.connect(self.ui.pbManualDiff, QtCore.SIGNAL("clicked()"), self.endDiffManual) 
        
        
        self.ui.statusbar.showMessage( "STARTED MANUAL-ASSIST DIFFRACTION SERIES" )
        
    def endDiffManual( self ):
        self.saveData( "DL_manual", self.DL_manual )
        self.hdf5handle.close()
        # Swap button functionality
        self.ui.pbManualDiff.setText( "Manual Diff START" )
        self.ui.pbManualDiff.setChecked( False )
        self.app.processEvents()
        QtCore.QObject.connect(self.ui.pbManualDiff, QtCore.SIGNAL("clicked()"), self.endDiffManual) 
        QtCore.QObject.connect(self.ui.pbManualDiff, QtCore.SIGNAL("clicked()"), self.startDiffManual) 
        
        self.ui.statusbar.showMessage( "FINISHED MANUAL-ASSIST DIFFRACTION SERIES" )
        
    def setIntensityLabel( self, newIntensity ):
        self.ui.labelIntensity.setText( "(" + str( self.int_counter+1 ) + ") Intensity: " + str( newIntensity ) )
        print str(self.int_counter) + ": ds focus = " + str( self.diffshift_focus[self.int_counter,:] )
        print str(self.int_counter) + ": ds aperture = " + str( self.diffshift_aperture[self.int_counter,:] )
        
        if( all(self.diffshift_focus[self.int_counter,:] != 999 )  ):
            self.ui.pbSetDLFocus.setChecked( True )
        else:
            self.ui.pbSetDLFocus.setChecked( False )
        if( all(self.diffshift_aperture[self.int_counter,:] != 999 ) ):
            self.ui.pbSetDLAperture.setChecked( True )
        else:
            self.ui.pbSetDLAperture.setChecked( False )    
        self.app.processEvents()
        
    def prevIntensity( self ):
        self.int_counter -= 1
        if self.int_counter < 0 :
            self.int_counter = 0
        
        self.tem.setLens( "int", self.intensity_vector[self.int_counter] )
        if self.last_set_focus:
            if( all(self.diffshift_focus[self.int_counter,:] != 999 ) ):
                self.tem.setDeflector( "ds",  self.diffshift_focus[self.int_counter,:] )
        else:
            if( all(self.diffshift_aperture[self.int_counter,:] != 999 ) ):
                self.tem.setDeflector( "ds",  self.diffshift_aperture[self.int_counter,:] )
                
        self.setIntensityLabel( self.intensity_vector[self.int_counter] )
        
        
    def nextIntensity( self ):
        self.int_counter += 1
        if self.int_counter > np.size( self.intensity_vector ) :
            self.int_counter = np.size( self.intensity_vector )
            
        self.tem.setLens( "int", self.intensity_vector[self.int_counter] )
        if self.last_set_focus:
            if( all(self.diffshift_focus[self.int_counter,:] != 999 ) ):
                self.tem.setDeflector( "ds", self.diffshift_focus[self.int_counter,:] )
        else:
            if( all(self.diffshift_aperture[self.int_counter,:] != 999 ) ):
                self.tem.setDeflector( "ds",  self.diffshift_aperture[self.int_counter,:] )
        self.setIntensityLabel( self.intensity_vector[self.int_counter] )
        
    def setVacuumArea( self ):
        self.position_vacuum = self.tem.getStage( "xy" )
        self.ui.statusbar.showMessage( "Updated vacuum position: "  + str(self.position_vacuum)  )
        
    def setCrystalArea( self ):
        self.position_crystal = self.tem.getStage( "xy")
        self.ui.statusbar.showMessage( "Updated crystal position: "  + str(self.position_vacuum)  )
        
    def gotoVacuumArea( self ):
        # if position is [0,0] do nothing
        if self.position_vacuum is [0,0]:
            self.ui.statusbar.showMessage( "No vacuum position set! " )
            return
        
        self.tem.setStage( "xy", self.position_vacuum )
        self.ui.statusbar.showMessage( "Goto vacuum position: "  + str(self.position_vacuum)  )        
        
    def gotoCrystalArea( self ):
        # if position is [0,0] do nothing
        if self.position_crystal is [0,0]:
            self.ui.statusbar.showMessage( "No crystal position set! " )
            return
        
        self.tem.setStage( "xy", self.position_crystal )
        self.ui.statusbar.showMessage( "Goto crystal position: "  + str(self.position_crystal)  )  
        
    def setDLFocus( self ):
        if self.manual_diff_init is False:
            return
            
        self.DL_manual[self.int_counter,0] = self.tem.getLens( "diff" )
        self.diffshift_focus[self.int_counter,:] = self.tem.getDeflector( "ds" )
        self.last_set_focus = True
        self.ui.statusbar.showMessage( "Focus DL registered at: " + str(self.DL_manual[self.int_counter,0]) + ", diff shift = " + str(self.diffshift_focus[self.int_counter,:]) )
        
        
    def setDLAperture( self ):
        if self.manual_diff_init is  False:
            return
            
        self.DL_manual[self.int_counter,1] = self.tem.getLens( "diff" )
        self.diffshift_aperture[self.int_counter,:] = self.tem.getDeflector( "ds" )
        self.last_set_focus = False
        self.ui.statusbar.showMessage( "Aperture DL registered at: " + str(self.DL_manual[self.int_counter,1]) + ", diff shift = " + str(self.diffshift_aperture[self.int_counter,:]) ) 

    def gotoDLFocus( self ):
        if self.manual_diff_init is False:
            return
        self.tem.setLens( "diff",  self.DL_manual[self.int_counter,0] )
        self.tem.setDeflector( "ds".  self.diffshift_focus[self.int_counter,:] )
        self.ui.statusbar.showMessage( "Goto Focus DL registered at: " + str(self.DL_manual[self.int_counter,0]) )
        
        
    def gotoDLAperture( self ):
        if self.manual_diff_init is False:
            return
        self.tem.setLens( "diff", self.DL_manual[self.int_counter,1] )
        self.tem.setDeflector( "ds",  self.diffshift_aperture[self.int_counter,:] )
        self.ui.statusbar.showMessage( "Goto Aperture DL registered at: " + str(self.DL_manual[self.int_counter,0]) )        
            
    def acquireVacuumDiff( self ):
        if self.manual_diff_init is False:
            self.ui.statusbar.showMessage( "Error: Manual-assisted series not started." )
            return
        self.ui.statusbar.showMessage( "Acquiring vacuum diffraction pair" )
        # Set to focus
        self.tem.setLens( "diff", self.DL_manual[self.int_counter,0] )
        self.tem.setDeflector( "ds",  self.diffshift_focus[self.int_counter,:] )
        
        # Calculate an appropriate exposure time
        binning_test = self.ui.sbTestBinning.value()
        tx_test = self.ui.sbTestTx.value()
        testdiff = self.tem.acquireImage( tx_test, binning_test )
        self.doselevel = np.max(testdiff)
        # Normalize to 1 second exposure equivalent
        self.doselevel = self.doselevel / tx_test / (binning_test/self.ui.sbBinning.value() )**2
        
        target_counts = self.ui.sbTargetCountsDiff.value()
        tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
        self.ui.statusbar.showMessage( "Vacuum focus diff tx = " + str(tx) + " s" )
        
        # Acquire images and write to the dset_vacuumfocus
        mage = np.zeros( [self.N, self.N] )
        for K in range(0,self.ui.sbFrameAverage.value() ):
            # TO DO: align the intermediate images with hyperspy?
            mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
        self.dset_vacuumfocus[:,:,self.int_counter] = mage
        
        #########################
        # Set to aperture
        self.tem.setLens( "diff", self.DL_manual[self.int_counter,1] )
        self.tem.setDeflector( "ds", self.diffshift_aperture[self.int_counter,:] )
        # Center and light meter
        
        # Calculate an appropriate exposure time
        binning_test = self.ui.sbTestBinning.value()
        tx_test = self.ui.sbTestTx.value()
        testdiff = self.tem.acquireImage( tx_test, binning_test )
        self.doselevel = np.max(testdiff)
        # Normalize to 1 second exposure equivalent
        self.doselevel = self.doselevel / tx_test / (binning_test/self.ui.sbBinning.value() )**2
        
        target_counts = self.ui.sbTargetCountsDiff.value()
        tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
        self.ui.statusbar.showMessage( "Vacuum aperture diff tx = " + str(tx) + " s" )
        
        # Acquire images and write to the dset_vacuumfocus
        mage = np.zeros( [self.N, self.N] )
        for K in range(0,self.ui.sbFrameAverage.value() ):
            # TO DO: align the intermediate images with hyperspy?
            mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
        self.dset_vacuumaperture[:,:,self.int_counter] = mage
            
        self.ui.statusbar.showMessage( "Finished vacuum diffraction pair for intensity #" + str(self.int_counter+1) )
        # end acquireVacuumDiff
        
    def acquireCrystalDiff( self ):
        if self.manual_diff_init is False:
            self.ui.statusbar.showMessage( "Error: Manual-assisted series not started." )
            return
        self.ui.statusbar.showMessage( "Acquiring crystal diffraction pair" )
        # Set to focus
        self.tem.setLens( "diff", self.DL_manual[self.int_counter,0] )
        self.tem.setDeflector( "ds", self.diffshift_focus[self.int_counter,:] )
        
        # Calculate an appropriate exposure time
        binning_test = self.ui.sbTestBinning.value()
        tx_test = self.ui.sbTestTx.value()
        testdiff = self.tem.acquireImage( tx_test, binning_test )
        self.doselevel = np.max(testdiff)
        # Normalize to 1 second exposure equivalent
        self.doselevel = self.doselevel / tx_test / (binning_test/self.ui.sbBinning.value() )**2
        
        target_counts = self.ui.sbTargetCountsDiff.value()
        tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
        self.ui.statusbar.showMessage( "Crystal focus diff tx = " + str(tx) + " s" )
        
        # Acquire images and write to the dset_crystalfocus
        mage = np.zeros( [self.N, self.N] )
        for K in range(0,self.ui.sbFrameAverage.value() ):
            # TO DO: align the intermediate images with hyperspy?
            mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
        self.dset_crystalfocus[:,:,self.int_counter] = mage
        
        #########################
        # Set to aperture
        self.tem.setLens( "diff", self.DL_manual[self.int_counter,1] )
        self.tem.setDeflector( "ds",  self.diffshift_aperture[self.int_counter,:] )
        # Center and light meter
        
        # Calculate an appropriate exposure time
        binning_test = self.ui.sbTestBinning.value()
        tx_test = self.ui.sbTestTx.value()
        testdiff = self.tem.acquireImage( tx_test, binning_test )
        self.doselevel = np.max(testdiff)
        # Normalize to 1 second exposure equivalent
        self.doselevel = self.doselevel / tx_test / (binning_test/self.ui.sbBinning.value() )**2
        
        target_counts = self.ui.sbTargetCountsDiff.value()
        tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
        self.ui.statusbar.showMessage( "Crystal aperture diff tx = " + str(tx) + " s" )
        
        # Acquire images and write to the dset_crystalfocus
        mage = np.zeros( [self.N, self.N] )
        for K in range(0,self.ui.sbFrameAverage.value() ):
            # TO DO: align the intermediate images with hyperspy?
            mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
        self.dset_crystalaperture[:,:,self.int_counter] = mage
            
        self.ui.statusbar.showMessage( "Finished crystal diffraction pair for intensity #" + str(self.int_counter+1) )
        # end acquirecrystalDiff
        
    def connectToTEM( self, temname ):
        # Connect to the FEITitan through the TNtem_feititan.py script
        if temname.lower() == 'feititan':
            self.tem = TEM.TEM_FeiTitan()
            self.tem_connected = self.tem.connectToTEM( self )
            if self.tem_connected is True:
                self.ui.statusbar.showMessage( "Successfully connected to FEI Titan" )
                # There doesn't seem to be an auto-exclude functionality as with radio buttons.
                self.ui.actionFEI_Titan.setChecked( True )
                self.ui.actionSimulator.setChecked( False )
                # self.reinitMicroscopeState()
                # self.ui.actionReInit.setEnabled( True )
            else:
                self.ui.statusbar.showMessage( "Failed to connect to FEI Titan" )
                self.ui.actionFEI_Titan.setChecked( False )
        if temname.lower() == 'simulator':
            self.tem = TEM.TEM_Simulator()
            self.tem_connected = self.tem.connectToTEM( self )
            self.ui.actionSimulator.setChecked( True )
            self.ui.actionFEI_Titan.setChecked( False )
            self.ui.statusbar.showMessage( "Connected to TEM simulator" )
            # self.reinitMicroscopeState()
            # self.ui.actionReInit.setEnabled( True )
    # end connectToTEM
        
    def setFileName( self ): 
        self.savefilename = str(QtGui.QFileDialog.getSaveFileName(self.MainWindow,"Save Data", "", "HDF5 Files (*.hdf5)"))
        print self.savefilename
        self.ui.labelSeriesFilename.setText( self.savefilename )
        
    def calibrateDescanDS( self ):
        # This function can be eliminated by using functools to link the event handler in the main func
        self.calibrateDescan( "ds" )
        

    
    def applyDescan( self, delta_nominal, deflector="ds" ):
        # We should have been passed a delta_nominal and now we just use it
        self.tem.setDeflector( deflector, delta_nominal )

            

    def acquireCaliSeries( self ):
        if self.tem_connected == False:
            self.ui.statusbar.showMessage( "Error: not connected to TEM" )
            return
            
        if self.savefilename is "":
            self.setFileName()
            
        # Find out if we are in probe or diff mode and call the appropriate sub-function
        if self.ui.rbProbeSeries.isChecked():
            self.acquireCaliProbe()
        elif self.ui.rbDiffSeries.isChecked():
            self.acquireCaliDiff()
    #end 
            
    def acquirePtycho( self ):
        if self.tem_connected == False:
            self.ui.statusbar.showMessage( "Error: not connected to TEM" )
            return
        
        if self.savefilename is "":
            self.setFileName()
            
        if self.ui.rbPtychoProbeScan.isChecked():
            self.acquirePtychoProbe()
        elif self.ui.rbPtychoDiffScan.isChecked():
            self.acquirePtychoDiff()
    # end
            

    def acquireCaliProbe( self ):
        """ This function is designed to capture the electron probe through the provided range of 
        Intensity values.  calibrateDescan is used to shift the probe back to center at each Intensity
        step.  These values should be saved into the descan_probe_temp registers at least...
        """
        
        # First check to see if we are using pre-existing descans or acquiring new ones.
        # This is important for series that have a specimen and cannot reliably be moved 
        # back to center...
        if self.ui.cbLoadOffsetsProbe.isChecked():
            self.descan_probe_temp = self.descan_probe_saveable
        else:
            self.descan_probe_temp = np.zeros( [self.ui.sbIntensitySteps.value(), 2] )    
            
        # Make intensity step vector
        intensity_start = self.ui.sbIntensityStart.value()
        intensity_end = self.ui.sbIntensityEnd.value()
        msteps = self.ui.sbIntensitySteps.value()
        self.intensity_vector = np.linspace( intensity_start, intensity_end, msteps )
        
        intensity_init = self.tem.getLens( "int" )
        shift_init = self.tem.getDeflector( "bs" )
        
        fhand = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        # File compression is quite slow unfortunately
        # probe_dset = fhand.create_dataset( "probe_data", (self.N, self.N, msteps), chunks=(self.N, self.N, 1), compression="gzip" )
        probe_dset = fhand.create_dataset( "probe_data", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        for J in range(0,msteps):
            self.tem.setLens( "int", self.intensity_vector[J] )      
            print " J : " + str(J+1) + " of " + str(msteps)
            
            # Determine need to descan or use saved descans
            if self.ui.cbFindOffsetsProbe.isChecked():
                [self.descan_probe_temp[J], _] = self.calibrateDescan( "bs" )
            elif self.ui.cbLoadOffsetsProbe.isChecked():
                # Use the values loaded into descan_probe_temp
                if self.descan_probe_temp is not None:
                    self.applyDescan( self.descan_probe_temp[J], "bs" )
            
            # Here I could end the for loop and start a new one but I don't think it is necessary at present
            
            # Calculate an appropriate exposure time
            target_counts = self.ui.sbTargetCountsDiff.value()
            tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
            print "tx set to : " + str(tx)
            
            # Acquire a probe image
            mage = np.zeros( [self.N, self.N] )
            for K in range(0,self.ui.sbFrameAverage.value() ):
                # TO DO: align the intermediate images with hyperspy?
                mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
            probe_dset[:,:,J] = mage
        # Return to initial state
        
        fhand.close()
        
        # Return to original state
        self.tem.setLens( "int", intensity_init )
        self.tem.setDeflector( "bs", shift_init )
        
        # Save the series
        # self.saveData( "probe_data", probe_data )
        self.saveData( "probe_descans", self.descan_probe_temp )
        self.saveData("intensity_vector", self.intensity_vector )
        
        if( self.ui.cbCloseColumnValve.isChecked() ):
            self.tem.closeValves()
    #end acquireCaliProbe
    pass
        
    def acquireCaliDiff( self ):
        # There is a general assumption here that the user has already acquired a 
        # probe series.
    
        # First check to see if we are using pre-existing descans or acquiring new ones.
        # This is important for series that have a specimen and cannot reliably be moved 
        # back to center...
        if self.ui.cbLoadOffsetsDiff.isChecked():
            self.descan_diff_temp = self.descan_diff_saveable
        else:
            self.descan_diff_temp = np.zeros( [self.ui.sbIntensitySteps.value(), self.ui.sbDiffSteps.value(), 2] )    
        
        # Make intensity step vector
        intensity_start = self.ui.sbIntensityStart.value()
        intensity_end = self.ui.sbIntensityEnd.value()
        msteps = self.ui.sbIntensitySteps.value()
        self.intensity_vector = np.linspace( intensity_start, intensity_end, msteps )
        print "Intensity : " + str(self.intensity_vector)
        
        # Make a diffraction step vector
        defocus_start = self.ui.sbDiffStart.value() * 1E-6
        defocus_end = self.ui.sbDiffEnd.value() * 1E-6
        nsteps = self.ui.sbDiffSteps.value()
        self.defocus_vector = np.linspace( defocus_start, defocus_end, nsteps )
        print "Defocus : " + str(self.defocus_vector)
        
        intensity_init = self.tem.getLens( "int" )
        shift_init = self.tem.getDeflector( "bs" )
        defocus_init = self.tem.getLens( "diff" )
        
        fhand = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        # File compresion is slow...
        # diff_dset = fhand.create_dataset( "diff_data", (self.N, self.N, msteps, nsteps), chunks=(self.N, self.N, 1, 1), compression="gzip" )
        diff_dset = fhand.create_dataset( "diff_data", (self.N, self.N, msteps, nsteps), chunks=(self.N, self.N, 1, 1) )        
        for J in range(0,msteps):
            self.tem.setLens( "int", self.intensity_vector[J] )
            print " J : " + str(J+1) + " of " + str(msteps)
            
            for I in range(0,nsteps):
                self.tem.setLens( "diff", self.defocus_vector[I] )
                print " I : " + str(I+1) + " of " + str(nsteps)
                
                if I == 0 and J > 0:
                    print "Reseting decan to : " + str( self.descan_diff_temp[J-1,I] )
                    self.applyDescan( self.descan_diff_temp[J-1,I], "ds" )
                    
                # Determine need to descan or use saved descans
                if self.ui.cbFindOffsetsProbe.isChecked():
                    [self.descan_diff_temp[J,I], _] = self.calibrateDescan( "ds" )
                elif self.ui.cbLoadOffsetsProbe.isChecked():
                    # Use the values loaded into descan_probe_temp
                    if self.descan_probe_temp is not None:
                        self.applyDescan( self.descan_probe_temp[J,I], "ds" )
                
                # Here I could end the for loop and start a new one but I don't think it is necessary at present
                
                # Calculate an appropriate exposure time
                target_counts = self.ui.sbTargetCountsDiff.value()
                tx = min( [self.ui.sbMaxTx.value(), target_counts / self.doselevel ])
                print "tx set to : " + str(tx)
            
                # Acquire a diffractogram
                mage = np.zeros( [self.N, self.N] )
                for K in range(0,self.ui.sbFrameAverage.value() ):
                    # TO DO: align the intermediate images with hyperspy?
                    mage = mage + self.tem.acquireImage( tx, self.ui.sbBinning.value() )                
                diff_dset[:,:,J,I] = mage
        # Return to initial state
        fhand.close()
        self.tem.setLens( "int", intensity_init )
        self.tem.setDeflector( "ds", shift_init )
        self.tem.setLens( "diff", defocus_init )
        
                
        self.saveData( "diff_descans", self.descan_diff_temp )
        self.saveData( "intensity_vector", self.intensity_vector )
        self.saveData( "defocus_vector", self.defocus_vector)
        
        if( self.ui.cbCloseColumnValve.isChecked() ):
            self.tem.closeValves()
        #end acquireCaliDiff
        pass
        
    def storeProbeOffsets( self ):
        self.descan_probe_saveable = self.descan_probe_temp
        self.ui.cbFindOffsetsProbe.setChecked( False )
        pass
        
    def storeDiffOffsets( self ):
        self.descan_diff_saveable = self.descan_diff_temp
        self.ui.cbFindOffsetsDiff.setChecked( False )
        pass
    
    def saveData( self, dsetname, dsetdata ):
        # Open the HDF5 file and make a dset
        fhand = h5py.File( self.savefilename ,'a') # file handle, saves in same directory as script
        
        fhand.create_dataset( dsetname, data=dsetdata)
        
        print "TO DO: save microscope state"
        fhand.close()
        pass
        
    def gotoStart( self ):
        # Find out if we are in probe or diff mode and call the appropriate sub-function
        if self.ui.rbProbeSeries.isChecked():
            self.tem.setLens( "int",  self.ui.sbIntensityStart.value() )
        elif self.ui.rbDiffSeries.isChecked():
            self.tem.setLens( "int",  self.ui.sbIntensityStart.value() )
            self.tem.setLens( "diff",  self.ui.sbDiffStart.value()*1E-6 )
            pass
            
    def gotoEnd( self ):
        # Find out if we are in probe or diff mode and call the appropriate sub-function
        if self.ui.rbProbeSeries.isChecked():
            self.tem.setLens( "int",  self.ui.sbIntensityEnd.value() )
        elif self.ui.rbDiffSeries.isChecked():
            self.tem.setLens( "int",  self.ui.sbIntensityEnd.value() )
            self.tem.setLens( "diff",  self.ui.sbDiffEnd.value()*1E-6 )
            pass
            
    def ptychoGenerateCoords( self, pattern = 'notset'  ):
        stepsize = self.ui.sbScanStep.value() * 1e-9
        xsteps = self.ui.sbXScanCount.value()
        ysteps = self.ui.sbYScanCount.value()
        
        init_bs = self.tem.getDeflector( "bs" )
        print "init_bs" + str(init_bs)
        
        if pattern == 'notset':
            pattern = str(self.ui.cbScanPatType.currentText())
        
        pattern = pattern.lower()
        if pattern == 'hex':
            # Current scan position is in the middle, so scan out
            # We only use xstep for the hexagonal scan
            self.ptycho_bs_pos = ram.hexmesh( stepsize, xsteps ) 
        elif pattern == 'grid':
            self.ptycho_bs_pos = ram.gridmesh( stepsize, xsteps, ysteps ) 
        else:
            print "Warning: no scan type selected"
            self.ptycho_bs_pos = np.array( [0.0,0.0] )
            
        # Add the initial position to the scans
        self.ptycho_bs_pos[0,:] = self.ptycho_bs_pos[0,:] + init_bs[0]
        self.ptycho_bs_pos[1,:] = self.ptycho_bs_pos[1,:] + init_bs[1]
        
        # return self.ptycho_bs_pos
        pass
        
    def ptychoCalibrateDescan( self ):
        self.ui.statusbar.showMessage( "Calibrating descans for Ptychography" )
        
        
        if self.tem_connected == False:
            self.ui.statusbar.showMessage( "Error: not connected to TEM" )
            return
        
        self.ptychoGenerateCoords( pattern = 'Grid' )
        
        print " shape bs : " + str( np.shape(self.ptycho_bs_pos))
        
        # Use the same methodology for the calibrateDescan but we are inducing a beamshift and we 
        # want to find out how much descan we have to apply with ds to zero it.
        # First go to the first beam shift in the list from ptycho_bs_pos
        bs_origin = self.ptycho_bs_pos[:,0]
        self.tem.setDeflector( "bs", bs_origin )
        
        # First, center the probe with the diffraction shifts
        self.calibrateDescan( "ds" )
        delta_ds_origin = self.tem.getDeflector( "ds" )
        
        # Now apply the beam shift of of the maximum shift in x
        
        bs_maxx = np.max( self.ptycho_bs_pos[0,:])
        self.tem.setDeflector( "bs", [bs_maxx,self.ptycho_bs_pos[1,0]] )
        # Now find descan with the diffraction shifts
        delta_ds_maxx = self.calibrateDescan( "ds" )
        
        bs_maxy = np.max( self.ptycho_bs_pos[1,:])
        self.tem.setDeflector( "bs", [self.ptycho_bs_pos[0,0], bs_maxy] )
        # Now find descan with the diffraction shifts
        delta_ds_maxy = self.calibrateDescan( "ds" )        
        
        # Now apply to the 3rd last scan position
        
        print "bs_first : " + str( self.ptycho_bs_pos[:,0])
        print "delta_ds_origin : " + str( delta_ds_origin )
        
        print "bs_maxx : " + str( bs_maxx )
        print "delta_ds_maxx : " + str( delta_ds_maxx )
        
        print "bs_maxy : " + str( bs_maxy )
        print "delta_ds_maxy : " + str( delta_ds_maxy )  
        
        descanx = (delta_ds_maxx ) / (bs_maxx - bs_origin[0] )
        descany = (delta_ds_maxy ) / (bs_maxy - bs_origin[1] )
        
        print "descanx : " + str(descanx)
        print "descany : " + str(descany)
        
        msteps = np.size( self.ptycho_bs_pos, 1)
        self.ptycho_ds_pos = np.zeros( np.shape(self.ptycho_bs_pos) )
        for J in range(0,msteps):
            self.ptycho_ds_pos[:,J] = delta_ds_origin + descanx * self.ptycho_bs_pos[0,J] + descany * self.ptycho_bs_pos[1,J]
           
            
        print str( self.ptycho_ds_pos )
        print "Finished Ptychography descan calibration" 
        pass
    # end ptychoCalibrateDescan
            
    def acquirePtychoProbe( self ):
        # if self.ptycho_bs_pos is None:
        self.ptychoGenerateCoords()
            
        print "Starting Ptychography probe acquisition"
        # ptycho_bs_pos has shape [2, msteps]
        msteps = np.size( self.ptycho_bs_pos, 1 )
        
        init_bs = self.tem.getDeflector( "bs" )
        tx = self.ui.sbPtychoTx.value()
        
        fhand = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        dsetname = str(self.ui.leDSetName.text())
        try:
            ptychoprobe_dset = fhand.create_dataset( dsetname, (self.N/self.ui.sbBinning.value(), self.N/self.ui.sbBinning.value(), msteps), chunks=(self.N/self.ui.sbBinning.value(), self.N/self.ui.sbBinning.value(), 1) )
        except RuntimeError:
            print "Error could not create dataset name: ptychodiff" 
            return
            
        for J in range(0,msteps):
            print "Acquire : " + str(J+1) + " of " + str(msteps)
            self.tem.setDeflector( "bs", self.ptycho_bs_pos[:,J] + init_bs )
            
            ptychoprobe_dset[:,:,J] = self.tem.acquireImage( tx, self.ui.sbBinning.value() )

        
        magesum = np.zeros( [self.N,self.N])
        print " Debug stop "
        for J in range(0,msteps):
            print "Add : " + str(J)
            magesum += ptychoprobe_dset[:,:,J]
        plt.imshow(magesum)
        plt.show()
        
        fhand.close()
        
        self.tem.setDeflector( "bs", init_bs )
        self.saveData( "ptycho_bs_pos", self.ptycho_bs_pos )
        pass
        
       
    def acquirePtychoDiff( self ):
        # if self.ptycho_bs_pos is None:
        self.ptychoGenerateCoords()
        # TO DO: add descan check
            
        print "Starting Ptychography diffractogram acquisition"
        # ptycho_bs_pos has shape [2, msteps]
        msteps = np.size( self.ptycho_bs_pos, 1 )
        
        init_bs = self.tem.getDeflector( "bs" )
        init_ds = self.tem.getDeflector( "ds" )
        tx = self.ui.sbPtychoTx.value()
        
        fhand = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        dsetname = str(self.ui.leDSetName.text())
        try:
            ptychodiff_dset = fhand.create_dataset( dsetname, (self.N/self.ui.sbBinning.value(), self.N/self.ui.sbBinning.value(), msteps), chunks=(self.N/self.ui.sbBinning.value(), self.N/self.ui.sbBinning.value(), 1) )
        except RuntimeError:
            print "Error could not create dataset name: ptychodiff" 
            return
            
        for J in range(0,msteps):
            
            print "Acquire : " + str(J+1) + " of " + str(msteps)
            print " Setting Beamshift to : " + str(self.ptycho_bs_pos[:,J])
            self.tem.setDeflector( "bs", self.ptycho_bs_pos[:,J] + init_bs )
            # self.tem.setDiffractionShift( self.ptycho_ds_pos[:,J])
            self.descanBSwithDS( self.ptycho_bs_pos[:,J] )
            plt.pause(0.05) # settling time
            ptychodiff_dset[:,:,J] = self.tem.acquireImage( tx, self.ui.sbBinning.value() )
            
            self.tem.setDeflector( "ds", init_ds )
        
        """
        magesum = np.zeros( [self.N,self.N])
        print " Debug stop "
        for J in range(0,msteps):
            print "Add : " + str(J)
            magesum += ptychodiff_dset[:,:,J]
        plt.imshow(magesum)
        plt.show()
        """
        
        
        fhand.close()
        
        self.tem.setDeflector( "bs", init_bs )
        self.tem.setDeflector( "ds", init_ds)
        self.saveData( "ptycho_bs_pos", self.ptycho_bs_pos )
        # self.saveData( "ptycho_ds_pos", self.ptycho_ds_pos )
        print "Finished Ptychographic scan" 
        pass
        
        
    def BFCancel( self ):
        self.BF_sentinel = True
    # So I need to start using the other interface.  Have a new file?  
    def BFScan( self ):
        # Need to have beam shifts along perspective of person, not the x-y scan coils!        
        self.BF_sentinel = False
        t0 = time.time()
        fasttx = self.ui.sbFastTx.value()
        fastbin = self.ui.sbFastBinning.value()
        fastsize = str( self.ui.cbFastFrameSize.currentText() ).lower()
        if( fastsize == 'full' ):
            framesize = 0
        elif( fastsize == 'half' ):
            framesize = 1
        elif( fastsize == 'quarter' ):
            framesize = 2
        fastint = self.ui.sbFastIntRadius.value()
        
        # Build coordinates from the Control panel
        self.ptychoGenerateCoords( pattern = 'grid' ) 
        
        # Do fast scan, similar to acquirePtychoDiff above
        print "Starting fast precession brightfield map"
        # ptycho_bs_pos has shape [2, msteps]
        msteps = np.size( self.ptycho_bs_pos, 1 )
        init_bs = self.tem.getDeflector( "bs" )
        init_ds = self.tem.getDeflector( "ds" )
        
        # Do a fast and dirty descan estimate for x and y
        if str(self.ui.labelRotCali.text()).lower() == 'uncalibrated':
            self.rotationCalibration()
        print ' debug' 
        # Rotate coords by -self.rot_image so we have the same reference frame as the user
        # DEBUG: THIS IS MORE COMPLICATED THAN I THINK BECAUSE X AND Y SHIFTS ARE NOT EQUAL.  SO THE COORDINATE TRANSFORM
        # IS NOT STRAIGHT-FORWARD
        # self.ptycho_bs_pos = ram.rotatemesh( self.ptycho_bs_pos, -self.rot_image ) 
        
        # We will just use matplotlib to display the brightfield map
        # fig, (ax1, ax2) = plt.subplots( 1, 2, sharey=True )
        self.bfmatrix = np.zeros( [self.ui.sbXScanCount.value(), self.ui.sbYScanCount.value()])
        self.bfxmesh = None
        self.bfymesh = None
              
        
        # plt.imshow( self.bfmatrix )
        # TO DO: seperate self.ptycho_bs_pos[0,;] into an array of strings (i.e. a tuple)
        # # plt.xticks( self.ptycho_bs_pos[0,:], str(self.ptycho_bs_pos[0,:]))
        # plt.yticks( self.ptycho_bs_pos[1,:], str(self.ptycho_bs_pos[1,:]))
        # plt.show( block=False )
        self.tem.image_cnt = 0 # For simulator, reset the image count to zero
        self.ui_ptychoBF.progressBar.setMinimum( 0 )
        self.ui_ptychoBF.progressBar.setMaximum( msteps-1 )
        t2 = time.time()
        for J in range(0,msteps):
            if self.BF_sentinel:
                # Stop processing, button was pushed
                return
                
            self.app.processEvents() # Unfreeze the GUI
            print "Acquire : " + str(J+1) + " of " + str(msteps)
            
            # Apply beam shift
            self.tem.setDeflector( "bs", self.ptycho_bs_pos[:,J] + init_bs )
            
            # Apply descan
            self.descanBSwithDS( self.ptycho_bs_pos[:,J] )
            plt.pause(0.05) # settling time
            
            # Acquire fast image
            # Unprocessed would be a tiny bit faster, proc = 0
            diffmage = self.tem.acquireImage( tx = fasttx, binning = fastbin, framesize = framesize, proc = 1 )
            # Reset diffraction shift
            self.tem.setDeflector( "ds", init_ds )
            
            # Localize the centroid of the diffraction pattern
            maxpos = np.unravel_index( np.argmax(diffmage), diffmage.shape )
            # And take a simple square sum
            # Could easily add a mask here if desired
            print "debug"
            indx = np.unravel_index( J, self.bfmatrix.shape )
            if J == 0:
                self.bfmatrix = np.ones( [self.ui.sbXScanCount.value(), self.ui.sbYScanCount.value()]) * diffmage[maxpos[0]-fastint:maxpos[0]+fastint, maxpos[1]-fastint:maxpos[1]+fastint].sum()
            else:
                self.bfmatrix[indx] = diffmage[maxpos[0]-fastint:maxpos[0]+fastint, maxpos[1]-fastint:maxpos[1]+fastint].sum()          
            
            self.app.processEvents() # Unfreeze the GUI
            # Try to dynamically show the brightfield image
            # plt.imshow( self.bfmatrix  )
            # plt.xticks( self.ptycho_bs_pos[0,:], str(self.ptycho_bs_pos[0,:]))
            # plt.yticks( self.ptycho_bs_pos[1,:], str(self.ptycho_bs_pos[1,:]))
            # pylab.axis( 'equal' )
            # plt.title( 'Diffraction brightfield scan image J = ' + str(J) + " of " + str(msteps)  )
            #diffplt = ax2.imshow( diffmage )
            #pylab.axis( 'equal' )
            #pylab.title( 'Diffractogram J = ' + str(J) + " of " + str(msteps)  )
            #diffplt.set_clim( ram.histClim(diffmage) )
            # Need to figure out how to dynamically update within a loop?
            # plt.draw()
            
            
            # Update BF display
            self.BFDraw()
            # Update status bar
            self.ui_ptychoBF.progressBar.setValue( J )
            plt.pause(0.01)
        t3 = time.time()
        #Reset shifts
        
        print "bfmatrix:" + str(self.bfmatrix)
        
        self.tem.setDeflector( "bs", init_bs )
        self.tem.setDeflector( "ds", init_ds )
        
        t1 = time.time()
        print "Time ellapsed : " + str( t1 - t0 ) + " s"
        print "Time per position : " + str( (t3 - t2)/msteps ) + " s"
        pass
        
        
    def BFInitCrosshair( self ):
        # Place a draggable cross-hairs on the scan window, that auto-updates to the 
        # two spin boxes
        print "TO DO: crosshair"
        pass
    
    def BFUpdateCrosshair( self ):
        print "TO DO: redraw crosshair"
        pass
    
    def BFDraw( self ):
        # Use ram.bilinear_interpolate(im, x, y) to upsample
        bf_qsize = self.ui_ptychoBF.graphicsView.size()
        
        if self.bfxmesh is None:
            self.bfxmesh, self.bfymesh = np.meshgrid( np.linspace( 0, self.bfmatrix.shape[1]-1,  bf_qsize.height()), np.linspace( 0, self.bfmatrix.shape[0]-1,  bf_qsize.width())  )  
        bf_display = ram.normalized( ram.bilinear_interpolate( self.bfmatrix, self.bfxmesh, self.bfymesh ) )
            
        print  "TO DO: remove xmesh if the window is resized, or if the scan size is changed"
        print "bf.min = " + str(bf_display.min()) + ", bf.max = " + str(bf_display.max())
        
        # Convert from numpy array to QImage
        self.local_image = q2np.gray2qimage( bf_display, cutoff= 0.01 )
        self.local_scene = QtGui.QGraphicsScene() 
        self.pixMapItem = QtGui.QGraphicsPixmapItem(QtGui.QPixmap(self.local_image), None, self.local_scene)
        self.ui_ptychoBF.graphicsView.setScene( self.local_scene )

        pass
    
    def BFGoTo( self ):
        # Go to the position indicated by the spinboxes
        gotopos = np.array( [self.ui_ptychoBF.sbBS_X.value(), self.ui_ptychoBF.sbBS_Y.value()] ) * 1.0E-6
        self.tem.setDeflector( "bs", gotopos )
        print "CHECK ME: BFGoTo.descanBSwithDS normally needs the DS to be reset between each call"
        self.descanBSwithDS( gotopos ) 
        self.BFUpdateCrosshair()
        pass
    
    def BFRefresh( self ):
        bs = self.tem.getDeflector( "bs" )
        self.ui_ptychoBF.sbBS_X.setValue( bs[0] * 1E6 )
        self.ui_ptychoBF.sbBS_X.setValue( bs[1] * 1E6 )
        self.BFUpdateCrosshair()
        pass
    
    """
    def acquirePtychoDiff( self ):
        if self.ptycho_bs_pos is None:
            self.ptychoGenerateCoords()
        # TO DO: add descan check
            
        print "Starting Ptychography diffactogram acquisition"
        # ptycho_bs_pos has shape [2, msteps]
        msteps = np.size( self.ptycho_bs_pos, 1 )
        
        init_bs = self.tem.getDeflector( "bs" )
        init_ds = self.tem.getDeflector( "ds" )
        tx = self.ui.sbPtychoTx.value()
        
        fhand = h5py.File( self.savefilename, 'a') # file handle, saves in same directory as script
        ptychodiff_dset = fhand.create_dataset( "ptychodiff", (self.N, self.N, msteps), chunks=(self.N, self.N, 1) )
        for J in range(0,msteps):
            print "Acquire : " + str(J+1) + " of " + str(msteps)
            self.tem.setDeflector( "bs", self.ptycho_bs_pos[:,J] )
            self.tem.setDeflector( "ds", self.ptycho_ds_pos[:,J])
            
            ptychodiff_dset[:,:,J] = self.tem.acquireImage( tx, self.ui.sbBinning.value() )
        
        
        magesum = np.zeros( [self.N,self.N])
        print " Debug stop "
        for J in range(0,msteps):
            print "Add : " + str(J)
            magesum += ptychodiff_dset[:,:,J]
        plt.imshow(magesum)
        plt.show()
        
        
        fhand.close()
        
        self.tem.setDeflector( "bs", init_bs )
        self.tem.setDeflector( "ds", init_ds)
        self.saveData( "ptycho_bs_pos", self.ptycho_bs_pos )
        self.saveData( "ptycho_ds_pos", self.ptycho_ds_pos )
        """
        
# end class PtychoControl

pc = PtychoControl()