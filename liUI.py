# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\GitHub\PtychoControl\liUI.ui'
#
# Created: Fri Apr 10 14:42:49 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_IsoplanWindow(object):
    def setupUi(self, IsoplanWindow):
        IsoplanWindow.setObjectName(_fromUtf8("IsoplanWindow"))
        IsoplanWindow.resize(1025, 1092)
        self.centralwidget = QtGui.QWidget(IsoplanWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.sbExposureTime = QtGui.QDoubleSpinBox(self.centralwidget)
        self.sbExposureTime.setGeometry(QtCore.QRect(100, 0, 62, 22))
        self.sbExposureTime.setProperty("value", 1.0)
        self.sbExposureTime.setObjectName(_fromUtf8("sbExposureTime"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 91, 21))
        self.label.setObjectName(_fromUtf8("label"))
        self.pbLiveIso = QtGui.QPushButton(self.centralwidget)
        self.pbLiveIso.setGeometry(QtCore.QRect(400, 0, 75, 23))
        self.pbLiveIso.setCheckable(True)
        self.pbLiveIso.setChecked(False)
        self.pbLiveIso.setObjectName(_fromUtf8("pbLiveIso"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 0, 46, 20))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.sbBinning = QtGui.QSpinBox(self.centralwidget)
        self.sbBinning.setGeometry(QtCore.QRect(210, 0, 42, 22))
        self.sbBinning.setMinimum(1)
        self.sbBinning.setMaximum(8)
        self.sbBinning.setObjectName(_fromUtf8("sbBinning"))
        self.labelPlot = QtGui.QLabel(self.centralwidget)
        self.labelPlot.setGeometry(QtCore.QRect(0, 30, 1024, 1024))
        self.labelPlot.setObjectName(_fromUtf8("labelPlot"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(270, 0, 61, 21))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.sbNoTiles = QtGui.QSpinBox(self.centralwidget)
        self.sbNoTiles.setGeometry(QtCore.QRect(330, 0, 42, 21))
        self.sbNoTiles.setMinimum(2)
        self.sbNoTiles.setProperty("value", 4)
        self.sbNoTiles.setObjectName(_fromUtf8("sbNoTiles"))
        IsoplanWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(IsoplanWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuMicroscope = QtGui.QMenu(self.menubar)
        self.menuMicroscope.setObjectName(_fromUtf8("menuMicroscope"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        IsoplanWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(IsoplanWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        IsoplanWindow.setStatusBar(self.statusbar)
        self.actionSimulator = QtGui.QAction(IsoplanWindow)
        self.actionSimulator.setCheckable(True)
        self.actionSimulator.setObjectName(_fromUtf8("actionSimulator"))
        self.actionFEI_Titan = QtGui.QAction(IsoplanWindow)
        self.actionFEI_Titan.setCheckable(True)
        self.actionFEI_Titan.setObjectName(_fromUtf8("actionFEI_Titan"))
        self.menuMicroscope.addAction(self.actionSimulator)
        self.menuMicroscope.addAction(self.actionFEI_Titan)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMicroscope.menuAction())

        self.retranslateUi(IsoplanWindow)
        QtCore.QMetaObject.connectSlotsByName(IsoplanWindow)

    def retranslateUi(self, IsoplanWindow):
        IsoplanWindow.setWindowTitle(_translate("IsoplanWindow", "Live Isoplanicity", None))
        self.label.setText(_translate("IsoplanWindow", "Exposure time (s):", None))
        self.pbLiveIso.setText(_translate("IsoplanWindow", "LiveIso", None))
        self.label_2.setText(_translate("IsoplanWindow", "Binning:", None))
        self.labelPlot.setText(_translate("IsoplanWindow", "labelPlot", None))
        self.label_3.setText(_translate("IsoplanWindow", "No of Tiles:", None))
        self.menuMicroscope.setTitle(_translate("IsoplanWindow", "Microscope", None))
        self.menuFile.setTitle(_translate("IsoplanWindow", "File", None))
        self.actionSimulator.setText(_translate("IsoplanWindow", "Simulator", None))
        self.actionFEI_Titan.setText(_translate("IsoplanWindow", "FEI Titan", None))

