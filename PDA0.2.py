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
import ConfigParser


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s
    
class PersistantDirectAligns:
    
    def __init__( self ):
        # VERBOSE declarations
        self.app = None
        self.MainWindow = None
        self.ui = None
        self.tem = None
        self.tem_connected = False
        
        self.app = QtGui.QApplication(sys.argv)
        self.MainWindow = QtGui.QMainWindow()

        self.ui = pdaUI.Ui_DeflectorWindow()
        self.ui.setupUi(self.MainWindow)
        
        self.ui.statusbar.showMessage( "Welcome to Personal Direct Alignments 0.1" )
        
        self.ui.actionFEI_Titan.triggered.connect( functools.partial( self.connectToTEM, 'FEITitan' ) )
        self.ui.actionSimulator.triggered.connect( functools.partial( self.connectToTEM, 'Simulator' )  )
        self.ui.actionReInit.triggered.connect( self.reinitMicroscopeState )
 
        self.ui.rbGS.clicked.connect( functools.partial( self.setActiveMode, "GS" ) )
        self.ui.rbGT.clicked.connect( functools.partial( self.setActiveMode, "GT" ) )
        self.ui.rbBS.clicked.connect( functools.partial( self.setActiveMode, "BS" ) )
        self.ui.rbBT.clicked.connect( functools.partial( self.setActiveMode, "BT" ) )
        self.ui.rbRC.clicked.connect( functools.partial( self.setActiveMode, "RC" ) )
        self.ui.rbIBS.clicked.connect( functools.partial( self.setActiveMode, "IBS" ) )
        self.ui.rbIBT.clicked.connect( functools.partial( self.setActiveMode, "IBT" ) )
        self.ui.rbIMS.clicked.connect( functools.partial( self.setActiveMode, "IMS" ) )
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
        self.modegroup.addButton( self.ui.rbIMS )
        self.modegroup.addButton( self.ui.rbDS ) 
        self.modegroup.addButton( self.ui.rbStage )

        # Re-direct keyPress functions
        self.ui.centralwidget.keyPressEvent = self.grabKey
        # Run through all the widgets and redirect the key-presses to everything BUT the spinboxes
        widgetlist = self.ui.centralwidget.findChildren( QtGui.QWidget )        
        print "TO DO: release focus from spinboxes on ENTER key press" 
        for mywidget in widgetlist:
            # print "Pause" 
            if not mywidget.__class__ is QtGui.QDoubleSpinBox:
                mywidget.keyPressEvent = self.grabKey

        # valueChanged fires whenever .setValue() is called, so use self.setSBBlocked(...) to 
        # update the slider without telling the microscope to change as well
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
        self.ui.sbIMS_X.valueChanged.connect( functools.partial( self.updateState, "IMS_X" ) )
        self.ui.sbIMS_Y.valueChanged.connect( functools.partial( self.updateState, "IMS_Y" ) )
        self.ui.sbDS_X.valueChanged.connect( functools.partial( self.updateState, "DS_X" ) )
        self.ui.sbDS_Y.valueChanged.connect( functools.partial( self.updateState, "DS_Y" ) )
        self.ui.sbStage_X.valueChanged.connect( functools.partial( self.updateState, "Stage_X" ) )
        self.ui.sbStage_Y.valueChanged.connect( functools.partial( self.updateState, "Stage_Y" ) )
        self.ui.sbStage_Z.valueChanged.connect( functools.partial( self.updateState, "Stage_Z" ) )
        self.ui.sbStage_Alpha.valueChanged.connect( functools.partial( self.updateState, "Stage_Alpha" ) )
        self.ui.sbStage_Beta.valueChanged.connect( functools.partial( self.updateState, "Stage_Beta" ) )
        self.ui.sbStepX.valueChanged.connect( functools.partial( self.updateStepSize, 'X' ) )
        self.ui.sbStepY.valueChanged.connect( functools.partial( self.updateStepSize, 'Y' ) )
        self.ui.sbStepZ.valueChanged.connect( functools.partial( self.updateStepSize, 'Z' ) )
        
        self.ui.SliderX.setScale( -100.0, 100.0 )
        self.ui.SliderX.setRange( -100.0, 100.0 )
        self.ui.SliderX.setStep( 1.0 )
        self.ui.SliderX.sliderMoved.connect( functools.partial( self.updateSlider, "X" ) )
        
        self.ui.SliderY.setScale( -100.0, 100.0 )
        self.ui.SliderY.setRange( -100.0, 100.0 )
        self.ui.SliderZ.setStep( 1.0 )
        self.ui.SliderY.sliderMoved.connect( functools.partial( self.updateSlider, "Y" ) )
        
        self.ui.SliderZ.setScale( -100.0, 100.0 )
        self.ui.SliderZ.setRange( -100.0, 100.0 )
        self.ui.SliderZ.setStep( 1.0 )
        self.ui.SliderZ.sliderMoved.connect( functools.partial( self.updateSlider, "Z" ) )
        
        # TO DO: implement a main loop to continuously update the GUI?  See how long a re-init takes from the real
        # microscope first!  
        
        self.MainWindow.show() 
        sys.exit(self.app.exec_())
        # end __init__
    
    def grabKey( self, event ):
        if( event.key() == QtCore.Qt.Key_Down ):
            updown = -1.0
            keyaxis = 'Y'
        elif( event.key() == QtCore.Qt.Key_Up ):
            updown = 1.0
            keyaxis = 'Y'
        elif( event.key() == QtCore.Qt.Key_Left ):
            updown = -1.0
            keyaxis = 'X'
        elif( event.key() == QtCore.Qt.Key_Right ):
            updown = 1.0
            keyaxis = 'X'
        elif( event.key() == QtCore.Qt.Key_PageUp ):
            updown = 1.0
            keyaxis = 'Z'
        elif( event.key() == QtCore.Qt.Key_PageDown ):
            updown = -1.0
            keyaxis = 'Z'    
        else:
            return
            
        activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_' + keyaxis
        # Now, how to call something based on this string...
        activespinbox = getattr( self.ui, "sb" + activemode )
        activespinbox.setValue( activespinbox.value() + updown * activespinbox.singleStep() )
        
        # update the sliders
        activeslider = getattr( self.ui, "Slider" + keyaxis )
        activeslider.setValue( activespinbox.value() )
        
    def setSBBlocked( self, identifier, newval ):
        # identifier = identifier.upper() # Stage isn't in caps

        # update the value
        if newval.size == 1:
            selectedsb = getattr( self.ui, "sb" + identifier )
            selectedsb.blockSignals( True )
            selectedsb.setValue( newval )
            selectedsb.blockSignals( False )
        elif newval.size == 2:
            # X-axis
            selectedsb = getattr( self.ui, "sb" + identifier + "_X" )
            selectedsb.blockSignals( True )
            selectedsb.setValue( newval[0] )
            selectedsb.blockSignals( False )
            # Y-axis
            selectedsb = getattr( self.ui, "sb" + identifier + "_Y" )
            selectedsb.blockSignals( True )
            selectedsb.setValue( newval[1] )
            selectedsb.blockSignals( False )
        
            
        
    def reinitMicroscopeState( self ):
        if self.tem_connected is False:
            return
            
        # Get and update limits for the spinboxes
        self.ui.sbStage_X.setMinimum( self.tem.lim_stage[0][1] * 1E9 )
        self.ui.sbStage_X.setMaximum( self.tem.lim_stage[0][0] * 1E9  )
        self.ui.sbStage_Y.setMinimum( self.tem.lim_stage[1][1] * 1E9  )
        self.ui.sbStage_Y.setMaximum( self.tem.lim_stage[1][0] * 1E9  )
        self.ui.sbStage_Z.setMinimum( self.tem.lim_stage[2][1] * 1E9  )
        self.ui.sbStage_Z.setMaximum( self.tem.lim_stage[2][0] * 1E9  )
        
        self.ui.sbStage_Alpha.setMinimum( self.tem.lim_stage[3][1] * 180 / np.pi  )
        self.ui.sbStage_Alpha.setMaximum( self.tem.lim_stage[3][0] * 180 / np.pi )
        self.ui.sbStage_Beta.setMinimum( self.tem.lim_stage[4][1] * 180 / np.pi )
        self.ui.sbStage_Beta.setMaximum( self.tem.lim_stage[4][0] * 180 / np.pi )
        
        self.ui.sbGS_X.setMinimum( self.tem.lim_gs[0][0] * 1E2 )
        self.ui.sbGS_X.setMaximum( self.tem.lim_gs[0][1] * 1E2 )
        self.ui.sbGS_Y.setMinimum( self.tem.lim_gs[1][0] * 1E2 )
        self.ui.sbGS_Y.setMaximum( self.tem.lim_gs[1][1] * 1E2 )
        
        self.ui.sbGT_X.setMinimum( self.tem.lim_gt[0][0] * 1E2 )
        self.ui.sbGT_X.setMaximum( self.tem.lim_gt[0][1] * 1E2 )
        self.ui.sbGT_Y.setMinimum( self.tem.lim_gt[1][0] * 1E2 )
        self.ui.sbGT_Y.setMaximum( self.tem.lim_gt[1][1] * 1E2 )
        
        self.ui.sbBS_X.setMinimum( self.tem.lim_bs[0][0] * 1E9 )
        self.ui.sbBS_X.setMaximum( self.tem.lim_bs[0][1] * 1E9)
        self.ui.sbBS_Y.setMinimum( self.tem.lim_bs[1][0] * 1E9)
        self.ui.sbBS_Y.setMaximum( self.tem.lim_bs[1][1] * 1E9)
        
        self.ui.sbBT_X.setMinimum( self.tem.lim_bt[0][0] * 1E3)
        self.ui.sbBT_X.setMaximum( self.tem.lim_bt[0][1] * 1E3)
        self.ui.sbBT_Y.setMinimum( self.tem.lim_bt[1][0] * 1E3)
        self.ui.sbBT_Y.setMaximum( self.tem.lim_bt[1][1] * 1E3)
        
        self.ui.sbRC_X.setMinimum( self.tem.lim_rc[0][0] * 1E3)
        self.ui.sbRC_X.setMaximum( self.tem.lim_rc[0][1] * 1E3)
        self.ui.sbRC_Y.setMinimum( self.tem.lim_rc[1][0] * 1E3)
        self.ui.sbRC_Y.setMaximum( self.tem.lim_rc[1][1] * 1E3)
        
        self.ui.sbIBS_X.setMinimum( self.tem.lim_ibs[0][0] * 1E9)
        self.ui.sbIBS_X.setMaximum( self.tem.lim_ibs[0][1] * 1E9)
        self.ui.sbIBS_Y.setMinimum( self.tem.lim_ibs[1][0] * 1E9)
        self.ui.sbIBS_Y.setMaximum( self.tem.lim_ibs[1][1] * 1E9)
        
        self.ui.sbIBT_X.setMinimum( self.tem.lim_ibt[0][0] * 1E3)
        self.ui.sbIBT_X.setMaximum( self.tem.lim_ibt[0][1] * 1E3)
        self.ui.sbIBT_Y.setMinimum( self.tem.lim_ibt[1][0] * 1E3)
        self.ui.sbIBT_Y.setMaximum( self.tem.lim_ibt[1][1] * 1E3)
        
        self.ui.sbIMS_X.setMinimum( self.tem.lim_is[0][0] * 1E9)
        self.ui.sbIMS_X.setMaximum( self.tem.lim_is[0][1] * 1E9)
        self.ui.sbIMS_Y.setMinimum( self.tem.lim_is[1][0] * 1E9)
        self.ui.sbIMS_Y.setMaximum( self.tem.lim_is[1][1] * 1E9)
        
        self.ui.sbDS_X.setMinimum( self.tem.lim_ds[0][0] * 1E3)
        self.ui.sbDS_X.setMaximum( self.tem.lim_ds[0][1] * 1E3)
        self.ui.sbDS_Y.setMinimum( self.tem.lim_ds[1][0] * 1E3)
        self.ui.sbDS_Y.setMaximum( self.tem.lim_ds[1][1] * 1E3)
        
        # Grab everything from the microscope and update the spinboxes
        self.setSBBlocked( 'GS', self.tem.getDeflector( 'GS' )*1E2)
        self.setSBBlocked( 'GT', self.tem.getDeflector( 'GT' )*1E2)
        self.setSBBlocked( 'BS', self.tem.getDeflector( 'BS' )*1E9)
        self.setSBBlocked( 'BT', self.tem.getDeflector( 'BT' )* 1E3)
        self.setSBBlocked( 'IBS', self.tem.getDeflector( 'IBS' )*1E9)
        self.setSBBlocked( 'IBT', self.tem.getDeflector( 'IBT' )* 1E3)
        self.setSBBlocked( 'IMS', self.tem.getDeflector( 'IMS' )*1E9)
        self.setSBBlocked( 'DS', self.tem.getDeflector( 'DS' )* 1E3)
        

        stageall = self.tem.getStage( 'all' )
        self.setSBBlocked( 'Stage_X', stageall[0]*1E9 )
        self.setSBBlocked( 'Stage_Y', stageall[1]*1E9 )
        self.setSBBlocked( 'Stage_Z', stageall[2]*1E9 )
        self.setSBBlocked( 'Stage_Alpha', stageall[3]*180/np.pi )
        self.setSBBlocked( 'Stage_Beta', stageall[4]*180/np.pi )

        # Update based on the new data
        activebutton = self.modegroup.checkedButton()
        rbname = activebutton.objectName()
        self.setActiveMode( rbname.split( "rb" )[1] )
        
    def updateSlider( self, identifier ):
        # Build a string that identifies the spinbox
        activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_' + identifier   
        # and find the slider and spinbox
        activeslider = getattr( self.ui, "Slider" + identifier )
        activespinbox = getattr( self.ui, "sb" + activemode )
        # Presto, update!
        # (spinbox event updates the microscope)
        activespinbox.setValue( activeslider.value() )
            
    def setSliderBlocked( self, identifier, newval ):
        # We don't need to know the spinbox if we are blocked 
        activeslider = getattr( self.ui, "Slider" + identifier )
        activeslider.blockSignals( True )
        activeslider.setValue( newval )
        activeslider.blockSignals( False )
        
    def updateStepSize( self, identifier ):
        activemode = str(self.modegroup.checkedButton().objectName()[2:]) + '_' + identifier   
        activestepbox = getattr( self.ui, 'sbStep' + identifier )
        activespinbox = getattr( self.ui, "sb" + activemode )
        activespinbox.setSingleStep( activestepbox.value() )
        
    def updateState( self, identifier ):
        # print "Updating state for ..." + str(identifier)
        if self.tem_connected is False:
            self.ui.statusbar.showMessage( "Not connected to a TEM" )
            return
        

        selectedspinbox = getattr( self.ui, "sb" + identifier )
        # Parse split on the underscore 
        [id_type, id_coord] = identifier.split( '_' )

        if( id_type == 'Stage' ):
            if id_coord == 'A' or id_coord == 'B':
                unitscale = np.pi/180
            else:
                unitscale = 1E-9
                
            print "Update stage: " + identifier + ", value : " + str(selectedspinbox.value()*unitscale)
            self.tem.setStage( id_coord, selectedspinbox.value() * unitscale )
        else: # If not the stage, we have a deflector
            if id_type[0] == 'G': # gs and gt
                unitscale = 1E-2
            elif id_type == 'BS' or id_type == 'IBS' or id_type == 'IMS':
                unitscale = 1E-9
            else:
                unitscale = 1E-3 # all angles in mrad

            # print "Update deflector: " + identifier + ", value : " + str(selectedspinbox.value()*unitscale)
            self.tem.setDeflector( identifier, selectedspinbox.value()*unitscale )
            
        # if radiobutton is selected, update the slider 
        testradiobut = getattr( self.ui, "rb" + id_type )
        if testradiobut.isChecked():
            self.setSliderBlocked( id_coord, selectedspinbox.value() )
                
        
        
    def setActiveMode( self, identifier ):
        print "Activating ..." + str(identifier)
        # TO DO: update the values in the spinboxes and update the sliders and 
        # step size windows.
        # If Stage, enable Z.  If not stage, disable Z
        if identifier == 'Stage':
            self.ui.SliderZ.setEnabled( True )
            self.ui.sbStepZ.setEnabled( True )
            modeaxes = [ 'X', 'Y', 'Z' ]
            
        else:
            self.ui.SliderZ.isEnabled()
            self.ui.SliderZ.setEnabled( False )
            self.ui.sbStepZ.setEnabled( False )
            modeaxes = [ 'X', 'Y' ]
            
        # Update slider based on the selected 
        activemode = str(self.modegroup.checkedButton().objectName()[2:]) 
        for x in modeaxes:
            activespinbox = getattr( self.ui, "sb" + activemode + '_' + x )
            activeslider = getattr( self.ui, "Slider" + x )
            activeslider.setScale( activespinbox.minimum(), activespinbox.maximum() )
            activeslider.setRange( activespinbox.minimum(), activespinbox.maximum() )
            activeslider.setStep( activespinbox.singleStep() )
            self.setSliderBlocked( x, activespinbox.value() )
            getattr( self.ui, 'sbStep' + x ).setValue( activespinbox.singleStep() )
            
        
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
                self.reinitMicroscopeState()
                self.ui.actionReInit.setEnabled( True )
            else:
                self.ui.statusbar.showMessage( "Failed to connect to FEI Titan" )
                self.ui.actionFEI_Titan.setChecked( False )
        if temname.lower() == 'simulator':
            self.tem = TEM.TEM_Simulator()
            self.tem_connected = self.tem.connectToTEM( self )
            self.ui.actionSimulator.setChecked( True )
            self.ui.actionFEI_Titan.setChecked( False )
            self.ui.statusbar.showMessage( "Connected to TEM simulator" )
            self.reinitMicroscopeState()
            self.ui.actionReInit.setEnabled( True )
    # end connectToTEM
    

    
    def updateGui( self ):
        # Do nothing for now, Python and QT seem to do a good job of updating themselves.
        pass
        
    def loadINIFile( self, filename ):
        # TO DO Example code at present
        cp = ConfigParser.SafeConfigParser()
        cp.read( filename )
        camlen = np.zeros( [20, 1]) 
        camlenp[0] = cp.getfloat( 'CameraLength', '400mm' )
        
        # Learn how to parse a file Python style.
        
    def saveINIFile( self, filename ):
        # TO DO Example code at present
        cp = ConfigParser.SafeConfigParser()
        cp.add_section( 'CameraLength' )
        cp.set( 'CameraLength', '400mm', '0.386' )
        with open( filename, 'wb' ) as cffile:
            cp.write( cffile )
        
pda = PersistantDirectAligns()