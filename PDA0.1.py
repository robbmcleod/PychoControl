# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:27:06 2014

@author: Robert A. McLeod

Persistant Direct Alignments (PDA)

Designed to allow manupulation of the alignments on a FEI Titan-class instrument 
regardless of the limitations of the TEM Interface software package.
"""




import numpy as np
from PyQt4 import QtCore, QtGui
# import os
import sys
import TEM
import pdaUI
# import BeamControl as bc
# import matplotlib.pyplot as plt
# import h5py
# from skimage.feature import match_template
# import pylab
# import time
import functools

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s
    
class PersistantDirectAligns:
    
    def __init__( self ):
        self.app = None
        self.MainWindow = None
        self.ui = None
        self.tem_interface = None
        self.tem_connected = False
        
        self.app = QtGui.QApplication(sys.argv)
        self.MainWindow = QtGui.QMainWindow()

        self.ui = pdaUI.Ui_DeflectorWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.ui.statusbar.showMessage( "Welcome to Personal Direct Alignments 0.1" )
        
        self.ui.actionFEI_Titan.triggered.connect( self.connecttoTEM_FeiTitan )
        self.ui.actionSimulator.triggered.connect( self.connecttoTEM_Simulator )
        self.ui.actionReInit.triggered.connect( self.reinitMicroscopeState )
 
        self.ui.rbGS.clicked.connect( functools.partial( self.setActiveMode, "GS" ) )
        self.ui.rbGT.clicked.connect( functools.partial( self.setActiveMode, "GT" ) )
        self.ui.rbBS.clicked.connect( functools.partial( self.setActiveMode, "BS" ) )
        self.ui.rbBT.clicked.connect( functools.partial( self.setActiveMode, "BT" ) )
        self.ui.rbRC.clicked.connect( functools.partial( self.setActiveMode, "RC" ) )
        self.ui.rbIBS.clicked.connect( functools.partial( self.setActiveMode, "IBS" ) )
        self.ui.rbIBT.clicked.connect( functools.partial( self.setActiveMode, "IBT" ) )
        self.ui.rbIS.clicked.connect( functools.partial( self.setActiveMode, "IS" ) )
        self.ui.rbDS.clicked.connect( functools.partial( self.setActiveMode, "DS" ) )
        self.ui.rbStage.clicked.connect( functools.partial( self.setActiveMode, "Stage" ) )
        self.modegroup = QtGui.QButtonGroup()
        self.modegroup.addButton( self.ui.rbGS )
        self.modegroup.addButton( self.ui.rbGT )
        self.modegroup.addButton( self.ui.rbBS )
        self.modegroup.addButton( self.ui.rbBT )
        self.modegroup.addButton( self.ui.rbRC )
        self.modegroup.addButton( self.ui.rbIBS )
        self.modegroup.addButton( self.ui.rbIBT )
        self.modegroup.addButton( self.ui.rbIS )
        self.modegroup.addButton( self.ui.rbDS ) 
        self.modegroup.addButton( self.ui.rbStage )
        
        # Just a tuple to track the widgets
        self.stagegroup = [self.ui.sbStage_X, self.ui.sbStage_Y, self.ui.sbStage_Z, self.ui.sbStage_Alpha, self.ui.sbStage_Beta]
        self.deflectgroup = [self.ui.sbGS_X, self.ui.sbGS_Y, self.ui.sbGT_X, self.ui.sbGT_Y, 
                             self.ui.sbBS_X, self.ui.sbBS_Y, self.ui.sbBT_X, self.ui.sbBT_Y, 
                             self.ui.sbRC_X, self.ui.sbRC_Y,
                             self.ui.sbIBS_X, self.ui.sbIBS_Y, self.ui.sbIBT_X, self.ui.sbIBT_Y, 
                             self.ui.sbIS_X, self.ui.sbIS_Y, self.ui.sbDT_X, self.ui.sbDT_Y ]

        # Re-direct keyPress functions
        self.ui.centralwidget.keyPressEvent = self.grabKey
        # Can I run through all the children (recurisively) to do this for all of them?
        widgetlist = self.ui.centralwidget.findChildren( QtGui.QWidget )
        for mywidget in widgetlist:
            # print "Pause" 
            mywidget.keyPressEvent = self.grabKey

        
        
        self.ui.sbGS_X.valueChanged.connect( functools.partial( self.updateState, "GS_X" ) )
        self.ui.sbGS_Y.valueChanged.connect( functools.partial( self.updateState, "GS_Y" ) )
        self.ui.sbGT_X.valueChanged.connect( functools.partial( self.updateState, "GT_X" ) )
        self.ui.sbGT_Y.valueChanged.connect( functools.partial( self.updateState, "GT_Y" ) )
        self.ui.sbBS_X.valueChanged.connect( functools.partial( self.updateState, "BS_X" ) )
        self.ui.sbBS_Y.valueChanged.connect( functools.partial( self.updateState, "BS_Y" ) )
        self.ui.sbBT_X.valueChanged.connect( functools.partial( self.updateState, "BT_X" ) )
        self.ui.sbBT_Y.valueChanged.connect( functools.partial( self.updateState, "BT_Y" ) )
        self.ui.sbRC_X.valueChanged.connect( functools.partial( self.updateState, "RC_X" ) )
        self.ui.sbRC_Y.valueChanged.connect( functools.partial( self.updateState, "RC_Y" ) )
        self.ui.sbIBS_X.valueChanged.connect( functools.partial( self.updateState, "IBS_X" ) )
        self.ui.sbIBS_Y.valueChanged.connect( functools.partial( self.updateState, "IBS_Y" ) )
        self.ui.sbIBT_X.valueChanged.connect( functools.partial( self.updateState, "IBT_X" ) )
        self.ui.sbIBT_Y.valueChanged.connect( functools.partial( self.updateState, "IBT_Y" ) )
        self.ui.sbIS_X.valueChanged.connect( functools.partial( self.updateState, "IS_X" ) )
        self.ui.sbIS_Y.valueChanged.connect( functools.partial( self.updateState, "IS_Y" ) )
        self.ui.sbDS_X.valueChanged.connect( functools.partial( self.updateState, "DS_X" ) )
        self.ui.sbDS_Y.valueChanged.connect( functools.partial( self.updateState, "DS_Y" ) )
        self.ui.sbStage_X.valueChanged.connect( functools.partial( self.updateState, "Stage_X" ) )
        self.ui.sbStage_Y.valueChanged.connect( functools.partial( self.updateState, "Stage_Y" ) )
        self.ui.sbStage_Z.valueChanged.connect( functools.partial( self.updateState, "Stage_Z" ) )
        self.ui.sbStage_Alpha.valueChanged.connect( functools.partial( self.updateState, "Stage_Alpha" ) )
        self.ui.sbStage_Beta.valueChanged.connect( functools.partial( self.updateState, "Stage_Beta" ) )
        
        # self.ui.Slider.setMin
        
        self.ui.SliderX.setScale( -100.0, 100.0 )
        # self.ui.SliderX.setScalePosition( 0.0 )
        self.ui.SliderX.setRange( -100.0, 100.0 )
        self.ui.SliderX.setStep( 1.0 )
        self.ui.SliderX.sliderMoved.connect( functools.partial( self.updateSlider, "X" ) )
        # self.ui.SliderX.valueChanged.connect( 
        
        self.ui.SliderY.setScale( -100.0, 100.0 )
        self.ui.SliderY.setRange( -100.0, 100.0 )
        self.ui.SliderZ.setStep( 1.0 )
        self.ui.SliderY.sliderMoved.connect( functools.partial( self.updateSlider, "Y" ) )
        
        self.ui.SliderZ.setScale( -100.0, 100.0 )
        self.ui.SliderZ.setRange( -100.0, 100.0 )
        self.ui.SliderZ.setStep( 1.0 )
        self.ui.SliderZ.sliderMoved.connect( functools.partial( self.updateSlider, "Z" ) )
        
        # sbGS_X, sbGS_Y, 
        self.MainWindow.show() 
        sys.exit(self.app.exec_())
        # end __init__
    
    def grabKey( self, event ):
        print "Test"
        if( event.key() == QtCore.Qt.Key_Down ):
            updown = -1.0
            keyaxis = 'Y'
            activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_Y'
        elif( event.key() == QtCore.Qt.Key_Up ):
            updown = 1.0
            keyaxis = 'Y'
            activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_Y'
        elif( event.key() == QtCore.Qt.Key_Left ):
            updown = -1.0
            keyaxis = 'X'
            activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_X'
        elif( event.key() == QtCore.Qt.Key_Right ):
            updown = 1.0
            keyaxis = 'X'
            activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_X'
            
        # Now, how to call something based on this string...
        activespinbox = getattr( self.ui, "sb" + activemode )
        activespinbox.setValue( activespinbox.value() + updown * activespinbox.singleStep() )
        
        # update the sliders
        self.updateSlider( keyaxis )
        
    def reinitMicroscopeState( self ):
        if self.tem_connected is False:
            return
            
        # Grab everything from the microscope and update the spinboxes
        gs = self.tem_interface.getGunShift()
        self.ui.sbGS_X.setValue( gs[0] * 1e2 )
        self.ui.sbGS_Y.setValue( gs[1] * 1e2 )
        
        gt = self.tem_interface.getGunTilt()
        self.ui.sbGT_X.setValue( gt[0] * 1e2 )
        self.ui.sbGT_Y.setValue( gt[1] * 1e2 )
        
        bs = self.tem_interface.getBeamShift()
        self.ui.sbBS_X.setValue( bs[0] * 1e6)
        self.ui.sbBS_Y.setValue( bs[1] * 1e6 )
        
        bt = self.tem_interface.getBeamTilt()
        self.ui.sbBT_X.setValue( bt[0] * 1e3 )
        self.ui.sbBT_Y.setValue( bt[1] * 1e3 )
        
        ibs = self.tem_interface.getImageBeamShift()
        self.ui.sbIBS_X.setValue( ibs[0] * 1e6)
        self.ui.sbIBS_Y.setValue( ibs[1] * 1e6 )
        
        ibt = self.tem_interface.getImageBeamTilt()
        self.ui.sbIBT_X.setValue( ibt[0] * 1e3 )
        self.ui.sbIBT_Y.setValue( ibt[1] * 1e3 )
        
        isd = self.tem_interface.getImageShift()
        self.ui.sbIS_X.setValue( isd[0] * 1e6)
        self.ui.sbIS_Y.setValue( isd[1] * 1e6 )
        
        ds = self.tem_interface.getDiffractionShift()
        self.ui.sbDS_X.setValue( ds[0] * 1e3 )
        self.ui.sbDS_Y.setValue( ds[1] * 1e3 )
        
        stagexy = self.tem_interface.getStageXY()
        self.ui.sbStage_X.setValue( stagexy[0] * 1e6 )        
        self.ui.sbStage_Y.setValue( stagexy[1] * 1e6 )       
        stagez = self.tem_interface.getStageZ()
        self.ui.sbStage_Z.setValue( stagez * 1e6 )  
        stagetilt = self.tem_interface.getStageTilt()
        self.ui.sbStage_Alpha.setValue( stagetilt[0]  )        
        self.ui.sbStage_Beta.setValue( stagetilt[1] ) 
        
        # Get and update limits for the spinboxes
        self.ui.sbStage_X.setMinimum( self.tem_interface.lim_xyz[0][0] )
        self.ui.sbStage_X.setMaximum( self.tem_interface.lim_xyz[0][1] )
        self.ui.sbStage_Y.setMinimum( self.tem_interface.lim_xyz[1][0] )
        self.ui.sbStage_Y.setMaximum( self.tem_interface.lim_xyz[1][1] )
        self.ui.sbStage_Z.setMinimum( self.tem_interface.lim_xyz[2][0] )
        self.ui.sbStage_Z.setMaximum( self.tem_interface.lim_xyz[2][1] )
        
        self.ui.sbStage_Alpha.setMinimum( self.tem_interface.lim_tilt[0][0] )
        self.ui.sbStage_Alpha.setMaximum( self.tem_interface.lim_tilt[0][1] )
        self.ui.sbStage_Beta.setMinimum( self.tem_interface.lim_tilt[1][0] )
        self.ui.sbStage_Beta.setMaximum( self.tem_interface.lim_tilt[1][1] )
        
        self.ui.sbGS_X.setMinimum( self.tem_interface.lim_gs[0][0] )
        self.ui.sbGS_X.setMaximum( self.tem_interface.lim_gs[0][1] )
        self.ui.sbGS_Y.setMinimum( self.tem_interface.lim_gs[1][0] )
        self.ui.sbGS_Y.setMaximum( self.tem_interface.lim_gs[1][1] )
        
        self.ui.sbGT_X.setMinimum( self.tem_interface.lim_gt[0][0] )
        self.ui.sbGT_X.setMaximum( self.tem_interface.lim_gt[0][1] )
        self.ui.sbGT_Y.setMinimum( self.tem_interface.lim_gt[1][0] )
        self.ui.sbGT_Y.setMaximum( self.tem_interface.lim_gt[1][1] )
        
        self.ui.sbBS_X.setMinimum( self.tem_interface.lim_bs[0][0] )
        self.ui.sbBS_X.setMaximum( self.tem_interface.lim_bs[0][1] )
        self.ui.sbBS_Y.setMinimum( self.tem_interface.lim_bs[1][0] )
        self.ui.sbBS_Y.setMaximum( self.tem_interface.lim_bs[1][1] )
        
        self.ui.sbBT_X.setMinimum( self.tem_interface.lim_bt[0][0] )
        self.ui.sbBT_X.setMaximum( self.tem_interface.lim_bt[0][1] )
        self.ui.sbBT_Y.setMinimum( self.tem_interface.lim_bt[1][0] )
        self.ui.sbBT_Y.setMaximum( self.tem_interface.lim_bt[1][1] )
        
        self.ui.sbRC_X.setMinimum( self.tem_interface.lim_rc[0][0] )
        self.ui.sbRC_X.setMaximum( self.tem_interface.lim_rc[0][1] )
        self.ui.sbRC_Y.setMinimum( self.tem_interface.lim_rc[1][0] )
        self.ui.sbRC_Y.setMaximum( self.tem_interface.lim_rc[1][1] )
        
        self.ui.sbIBS_X.setMinimum( self.tem_interface.lim_ibs[0][0] )
        self.ui.sbIBS_X.setMaximum( self.tem_interface.lim_ibs[0][1] )
        self.ui.sbIBS_Y.setMinimum( self.tem_interface.lim_ibs[1][0] )
        self.ui.sbIBS_Y.setMaximum( self.tem_interface.lim_ibs[1][1] )
        
        self.ui.sbIBT_X.setMinimum( self.tem_interface.lim_ibt[0][0] )
        self.ui.sbIBT_X.setMaximum( self.tem_interface.lim_ibt[0][1] )
        self.ui.sbIBT_Y.setMinimum( self.tem_interface.lim_ibt[1][0] )
        self.ui.sbIBT_Y.setMaximum( self.tem_interface.lim_ibt[1][1] )
        
        self.ui.sbIS_X.setMinimum( self.tem_interface.lim_is[0][0] )
        self.ui.sbIS_X.setMaximum( self.tem_interface.lim_is[0][1] )
        self.ui.sbIS_Y.setMinimum( self.tem_interface.lim_is[1][0] )
        self.ui.sbIS_Y.setMaximum( self.tem_interface.lim_is[1][1] )
        
        self.ui.sbDS_X.setMinimum( self.tem_interface.lim_ds[0][0] )
        self.ui.sbDS_X.setMaximum( self.tem_interface.lim_ds[0][1] )
        self.ui.sbDS_Y.setMinimum( self.tem_interface.lim_ds[1][0] )
        self.ui.sbDS_Y.setMaximum( self.tem_interface.lim_ds[1][1] )
        
        # Play games to deselect all the buttons...
        activebutton = self.modegroup.checkedButton()
        rbname = activebutton.objectName()
        # print rbname.split( "rb" )[1]
        self.setActiveMode( rbname.split( "rb" )[1] )
        
    def updateSlider( self, identifier ):
        # print "Updated slider for : " + str(identifier)
        # print "Slider?.value : " + str(self.ui.SliderX.value()) + ", " + str(self.ui.SliderY.value()) + ", " + str(self.ui.SliderZ.value())
        
        # How to find which rb is enabled? 
        # Ergh... looping with findChildern impossible evidently...
        #activemod = self.modegroup.checkedButton()        
        
        if self.ui.rbStage.isChecked():
            if identifier == 'X':
                self.ui.sbStage_X.setValue( self.ui.SliderX.value())
                self.updateState( 'Stage_X' )
            elif identifier == 'Y':
                self.ui.sbStage_Y.setValue( self.ui.SliderY.value())  
                self.updateState( 'Stage_Y' )
            elif identifier == 'Z':
                self.ui.sbStage_Z.setValue( self.ui.SliderZ.value())   
                self.updateState( 'Stage_Z' )
        elif self.ui.rbGS.isChecked():
            if identifier == 'X':
                self.ui.sbGS_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbGS_Y.setValue( self.ui.SliderY.value())    
        elif self.ui.rbGT.isChecked():
            if identifier == 'X':
                self.ui.sbGT_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbGT_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbBS.isChecked():
            if identifier == 'X':
                self.ui.sbBS_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbBS_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbBT.isChecked():
            if identifier == 'X':
                self.ui.sbGT_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbGT_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbRC.isChecked():
            if identifier == 'X':
                self.ui.sbRC_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbRC_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbIBS.isChecked():
            if identifier == 'X':
                self.ui.sbIBS_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbIBS_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbIBT.isChecked():
            if identifier == 'X':
                self.ui.sbIBT_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbIBT_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbIS.isChecked():
            if identifier == 'X':
                self.ui.sbIS_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbIS_Y.setValue( self.ui.SliderY.value())   
        elif self.ui.rbDS.isChecked():
            if identifier == 'X':
                self.ui.sbDS_X.setValue( self.ui.SliderX.value())
            elif identifier == 'Y':
                self.ui.sbDS_Y.setValue( self.ui.SliderY.value())   
        
        
    def updateState( self, identifier ):
        # print "Updating state for ..." + str(identifier)
        if self.tem_connected is False:
            self.ui.statusbar.showMessage( "Not connected to a TEM" )
            return
        
        print "Updating state for ..." + str(identifier)
        print "TO DO: update to use getattr() and setattr()"
        if identifier == 'Stage_X':
            stage = [self.ui.sbStage_X.value(), np.NaN]
            self.tem_interface.setStageXY( stage )
            if self.ui.rbStage.isChecked():
                self.ui.SliderX.setValue( stage[0] )
        elif identifier == 'Stage_Y':
            stage = [np.NaN, self.ui.sbStage_Y.value()]
            self.tem_interface.setStageXY( stage )
            if self.ui.rbStage.isChecked():
                self.ui.SliderY.setValue( stage[1] )
        elif identifier == 'Stage_Z':
            stagez = self.ui.sbStage_Z.value()
            self.tem_interface.setStageZ( stagez )
            if self.ui.rbStage.isChecked():
                self.ui.SliderZ.setValue( stagez )
        elif identifier == 'Stage_Alpha':
            stagealpha = self.ui.sbStage_Alpha.value()
            self.tem_interface.setAlphaTilt( stagealpha )
        elif identifier == 'Stage_Beta':
            stagebeta = self.ui.sbStage_Beta.value()
            self.tem_interface.setBetaTilt( stagebeta )
        elif identifier == 'GS_X':
            gs = [self.ui.sbGS_X.value(), np.NaN]
            self.tem_interface.setGunShift( gs )
            if self.ui.rbGS.isChecked():
                self.ui.SliderX.setValue( gs[0])
        elif identifier == 'GS_Y':
            gs = [np.NaN, self.ui.sbGS_Y.value()]
            self.tem_interface.setGunShift( gs )
            if self.ui.rbGS.isChecked():
                self.ui.SliderY.setValue( gs[1])
        elif identifier == 'GT_X':
            gt = [self.ui.sbGT_X.value(), np.NaN]
            self.tem_interface.setGunTilt( gt )
            if self.ui.rbGT.isChecked():
                self.ui.SliderX.setValue( gt[0])
        elif identifier == 'GT_Y':
            gt = [np.NaN, self.ui.sbGT_Y.value()]
            self.tem_interface.setGunTilt( gt )
            if self.ui.rbGT.isChecked():
                self.ui.SliderY.setValue( gt[1])
        elif identifier == 'BS_X':
            bs = [self.ui.sbBS_X.value(), np.NaN]
            self.tem_interface.setBeamShift( bs )
            if self.ui.rbBS.isChecked():
                self.ui.SliderX.setValue( bs[0])
        elif identifier == 'BS_Y':
            bs = [np.NaN, self.ui.sbBS_Y.value()]
            self.tem_interface.setBeamShift( bs )
            if self.ui.rbBS.isChecked():
                self.ui.SliderY.setValue( bs[1])
        elif identifier == 'BT_X':
            bt = [self.ui.sbBT_X.value(), np.NaN]
            self.tem_interface.setBeamTilt( bt )
            if self.ui.rbBT.isChecked():
                self.ui.SliderX.setValue( bt[0])
        elif identifier == 'BT_Y':
            bt = [np.NaN, self.ui.sbBT_Y.value()]
            self.tem_interface.setBeamTilt( bt )
            if self.ui.rbBT.isChecked():
                self.ui.SliderY.setValue( bt[1])
        elif identifier == 'RC_X':
            rc = [self.ui.sbRC_X.value(), np.NaN]
            self.tem_interface.setRotationCenter( rc )
            if self.ui.rbRC.isChecked():
                self.ui.SliderX.setValue( rc[0])
        elif identifier == 'RC_Y':
            rc = [np.NaN, self.ui.sbRC_Y.value()]
            self.tem_interface.setRotationCenter( rc )
            if self.ui.rbRC.isChecked():
                self.ui.SliderY.setValue( rc[1])
        elif identifier == 'IBS_X':
            ibs = [self.ui.sbIBS_X.value(), np.NaN]
            self.tem_interface.setImageBeamShift( ibs )
            if self.ui.rbIBS.isChecked():
                self.ui.SliderX.setValue( ibs[0])
        elif identifier == 'IBS_Y':
            ibs = [np.NaN, self.ui.sbIBS_Y.value()]
            self.tem_interface.setImageBeamShift( ibs )
            if self.ui.rbIBS.isChecked():
                self.ui.SliderY.setValue( ibs[1])
        elif identifier == 'IBT_X':
            ibt = [self.ui.sbIBT_X.value(), np.NaN]
            self.tem_interface.setImageBeamTilt( ibt )
            if self.ui.rbIBT.isChecked():
                self.ui.SliderX.setValue( ibt[0])
        elif identifier == 'IBT_Y':
            ibt = [np.NaN, self.ui.sbIBT_Y.value()]
            self.tem_interface.setImageBeamTilt( ibt )
            if self.ui.rbIBT.isChecked():
                self.ui.SliderY.setValue( ibt[1])
        elif identifier == 'IS_X':
            ims = [self.ui.sbIS_X.value(), np.NaN]
            self.tem_interface.setImageShift( ims )
            if self.ui.rbIS.isChecked():
                self.ui.SliderX.setValue( ims[0])
        elif identifier == 'IS_Y':
            ims = [np.NaN, self.ui.sbIS_Y.value()]
            self.tem_interface.setImageShift( ims )
            if self.ui.rbIS.isChecked():
                self.ui.SliderY.setValue( ims[1])
        elif identifier == 'DS_X':
            ds = [self.ui.sbDS_X.value(), np.NaN]
            self.tem_interface.setDiffractionShift( ds )
            if self.ui.rbDS.isChecked():
                self.ui.SliderX.setValue( ds[0])
        elif identifier == 'DS_Y':
            ds = [np.NaN, self.ui.sbDS_Y.value()]
            self.tem_interface.setDiffractionShift( ds )
            if self.ui.rbDS.isChecked():
                self.ui.SliderY.setValue( ds[1])
        else:
            print "DEBUG, PDA.updateState()"
        
        
    def setActiveMode( self, identifier ):
        print "Activating ..." + str(identifier)
        # TO DO: update the values in the spinboxes and update the sliders and 
        # step size windows.
        # If Stage, enable Z.  If not stage, disable Z
        if identifier == 'Stage':
            self.ui.SliderZ.setEnabled( True )
            self.ui.sbStepZ.setEnabled( True )
            
        elif self.ui.SliderZ.isEnabled():
            self.ui.SliderZ.setEnabled( False )
            self.ui.sbStepZ.setEnabled( False )
            
        # Update slider based on the selected 
        # TO DO: re-write using setattr and getattr
        if identifier == 'Stage':
            self.ui.SliderX.setScale( self.ui.sbStage_X.minimum(), self.ui.sbStage_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbStage_X.minimum(), self.ui.sbStage_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbStage_X.value() )
            self.ui.SliderX.setStep( self.ui.sbStage_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbStage_Y.minimum(), self.ui.sbStage_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbStage_Y.minimum(), self.ui.sbStage_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbStage_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbStage_X.singleStep() )
            self.ui.SliderZ.setScale( self.ui.sbStage_Z.minimum(), self.ui.sbStage_Z.maximum() )
            self.ui.SliderZ.setRange( self.ui.sbStage_Z.minimum(), self.ui.sbStage_Z.maximum() )
            self.ui.SliderZ.setValue( self.ui.sbStage_Z.value() )
            self.ui.SliderZ.setStep( self.ui.sbStage_Z.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
            self.ui.sbStepZ.setValue( self.ui.SliderZ.step() )
        elif identifier == 'GS':
            self.ui.SliderX.setScale( self.ui.sbGS_X.minimum(), self.ui.sbGS_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbGS_X.minimum(), self.ui.sbGS_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbGS_X.value() )
            self.ui.SliderX.setStep( self.ui.sbGS_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbGS_Y.minimum(), self.ui.sbGS_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbGS_Y.minimum(), self.ui.sbGS_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbGS_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbGS_Y.singleStep() ) 
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'GT':
            self.ui.SliderX.setScale( self.ui.sbGT_X.minimum(), self.ui.sbGT_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbGT_X.minimum(), self.ui.sbGT_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbGT_X.value() )
            self.ui.SliderX.setStep( self.ui.sbGT_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbGT_Y.minimum(), self.ui.sbGT_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbGT_Y.minimum(), self.ui.sbGT_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbGT_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbGT_Y.singleStep() ) 
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'BS':
            self.ui.SliderX.setScale( self.ui.sbBS_X.minimum(), self.ui.sbBS_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbBS_X.minimum(), self.ui.sbBS_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbBS_X.value() )
            self.ui.SliderX.setStep( self.ui.sbBS_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbBS_Y.minimum(), self.ui.sbBS_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbBS_Y.minimum(), self.ui.sbBS_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbBS_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbBS_Y.singleStep() ) 
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'BT':
            self.ui.SliderX.setScale( self.ui.sbBT_X.minimum(), self.ui.sbBT_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbBT_X.minimum(), self.ui.sbBT_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbBT_X.value() )
            self.ui.SliderX.setStep( self.ui.sbBT_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbBT_Y.minimum(), self.ui.sbBT_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbBT_Y.minimum(), self.ui.sbBT_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbBT_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbBT_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'RC':
            self.ui.SliderX.setScale( self.ui.sbRC_X.minimum(), self.ui.sbRC_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbRC_X.minimum(), self.ui.sbRC_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbRC_X.value() )
            self.ui.SliderX.setStep( self.ui.sbRC_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbRC_Y.minimum(), self.ui.sbRC_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbRC_Y.minimum(), self.ui.sbRC_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbRC_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbRC_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'IBS':
            self.ui.SliderX.setScale( self.ui.sbIBS_X.minimum(), self.ui.sbIBS_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbIBS_X.minimum(), self.ui.sbIBS_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbIBS_X.value() )
            self.ui.SliderX.setStep( self.ui.sbIBS_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbIBS_Y.minimum(), self.ui.sbIBS_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbIBS_Y.minimum(), self.ui.sbIBS_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbIBS_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbIBS_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'IBT':
            self.ui.SliderX.setScale( self.ui.sbIBT_X.minimum(), self.ui.sbIBT_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbIBT_X.minimum(), self.ui.sbIBT_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbIBT_X.value() )
            self.ui.SliderX.setStep( self.ui.sbIBT_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbIBT_Y.minimum(), self.ui.sbIBT_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbIBT_Y.minimum(), self.ui.sbIBT_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbIBT_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbIBT_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'IS':
            self.ui.SliderX.setScale( self.ui.sbIS_X.minimum(), self.ui.sbIS_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbIS_X.minimum(), self.ui.sbIS_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbIS_X.value() )
            self.ui.SliderX.setStep( self.ui.sbIS_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbIS_Y.minimum(), self.ui.sbIS_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbIS_Y.minimum(), self.ui.sbIS_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbIS_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbIS_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        elif identifier == 'DS':
            self.ui.SliderX.setScale( self.ui.sbDS_X.minimum(), self.ui.sbDS_X.maximum() )
            self.ui.SliderX.setRange( self.ui.sbDS_X.minimum(), self.ui.sbDS_X.maximum() )
            self.ui.SliderX.setValue( self.ui.sbDS_X.value() )
            self.ui.SliderX.setStep( self.ui.sbDS_X.singleStep() )
            self.ui.SliderY.setScale( self.ui.sbDS_Y.minimum(), self.ui.sbDS_Y.maximum() )
            self.ui.SliderY.setRange( self.ui.sbDS_Y.minimum(), self.ui.sbDS_Y.maximum() )
            self.ui.SliderY.setValue( self.ui.sbDS_Y.value() )
            self.ui.SliderY.setStep( self.ui.sbDS_Y.singleStep() )
            self.ui.sbStepX.setValue( self.ui.SliderX.step() )
            self.ui.sbStepY.setValue( self.ui.SliderY.step() )
        
        
    def connecttoTEM_FeiTitan( self ):
        # Connect to the FEITitan through the TNtem_feititan.py script
        self.tem_interface = TEM.TEM_FeiTitan()
        self.tem_connected = self.tem_interface.connectToTEM( self )
        if self.tem_connected is True:
            self.ui.statusbar.showMessage( "Successfully connected to FEI Titan" )
            # There doesn't seem to be an auto-exclude functionality as with radio buttons.
            self.ui.actionFEI_Titan.setChecked( True )
            self.ui.actionSimulator.setChecked( False )
            self.reinitMicroscopeState()
            self.ui.actionReInit.setEnabled( True )
        else:
            self.ui.statusbar.showMessage( "Failed to connect to FEI Titan" )
            self.ui.actionFEI_Titan.setChecked( False )
    # end checkTEM_FeiTitan
    
    def connecttoTEM_Simulator( self ):
        # This being an empty simulator, we can basically always connect to it
        self.tem_interface = TEM.TEM_Simulator()
        self.tem_connected = self.tem_interface.connectToTEM( self )
        self.ui.actionSimulator.setChecked( True )
        self.ui.actionFEI_Titan.setChecked( False )
        self.ui.statusbar.showMessage( "Connected to TEM simulator" )
        self.reinitMicroscopeState()
        self.ui.actionReInit.setEnabled( True )
    # end checkTEM_Simulator
    
    def updateGui( self ):
        # Do nothing for now, Python and QT seem to do a good job of updating themselves.
        True
        
pda = PersistantDirectAligns()