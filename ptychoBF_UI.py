# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\GitHub\PtychoControl\ptychoBF_UI.ui'
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

class Ui_PtychoBFDialog(object):
    def setupUi(self, PtychoBFDialog):
        PtychoBFDialog.setObjectName(_fromUtf8("PtychoBFDialog"))
        PtychoBFDialog.resize(1035, 545)
        self.progressBar = QtGui.QProgressBar(PtychoBFDialog)
        self.progressBar.setGeometry(QtCore.QRect(400, 10, 118, 16))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.label_2 = QtGui.QLabel(PtychoBFDialog)
        self.label_2.setGeometry(QtCore.QRect(140, 10, 51, 21))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(PtychoBFDialog)
        self.label_3.setGeometry(QtCore.QRect(0, 10, 51, 20))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.tbCrosshair = QtGui.QToolButton(PtychoBFDialog)
        self.tbCrosshair.setGeometry(QtCore.QRect(280, 0, 31, 31))
        self.tbCrosshair.setObjectName(_fromUtf8("tbCrosshair"))
        self.sbBS_X = QtGui.QDoubleSpinBox(PtychoBFDialog)
        self.sbBS_X.setGeometry(QtCore.QRect(50, 10, 81, 22))
        self.sbBS_X.setDecimals(1)
        self.sbBS_X.setMaximum(100000.0)
        self.sbBS_X.setProperty("value", 0.0)
        self.sbBS_X.setObjectName(_fromUtf8("sbBS_X"))
        self.sbBS_Y = QtGui.QDoubleSpinBox(PtychoBFDialog)
        self.sbBS_Y.setGeometry(QtCore.QRect(190, 10, 81, 22))
        self.sbBS_Y.setDecimals(1)
        self.sbBS_Y.setMaximum(100000.0)
        self.sbBS_Y.setObjectName(_fromUtf8("sbBS_Y"))
        self.tbGoto = QtGui.QToolButton(PtychoBFDialog)
        self.tbGoto.setGeometry(QtCore.QRect(310, 0, 31, 31))
        self.tbGoto.setObjectName(_fromUtf8("tbGoto"))
        self.tbRefresh = QtGui.QToolButton(PtychoBFDialog)
        self.tbRefresh.setGeometry(QtCore.QRect(340, 0, 31, 31))
        self.tbRefresh.setObjectName(_fromUtf8("tbRefresh"))
        self.graphicsView = QtGui.QGraphicsView(PtychoBFDialog)
        self.graphicsView.setGeometry(QtCore.QRect(0, 30, 512, 512))
        self.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.graphicsView_Diff = QtGui.QGraphicsView(PtychoBFDialog)
        self.graphicsView_Diff.setGeometry(QtCore.QRect(520, 30, 512, 512))
        self.graphicsView_Diff.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_Diff.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.graphicsView_Diff.setObjectName(_fromUtf8("graphicsView_Diff"))

        self.retranslateUi(PtychoBFDialog)
        QtCore.QMetaObject.connectSlotsByName(PtychoBFDialog)

    def retranslateUi(self, PtychoBFDialog):
        PtychoBFDialog.setWindowTitle(_translate("PtychoBFDialog", "Dialog", None))
        self.label_2.setText(_translate("PtychoBFDialog", "Position Y: ", None))
        self.label_3.setText(_translate("PtychoBFDialog", "Position X: ", None))
        self.tbCrosshair.setText(_translate("PtychoBFDialog", "X", None))
        self.tbGoto.setText(_translate("PtychoBFDialog", "Goto", None))
        self.tbRefresh.setText(_translate("PtychoBFDialog", "Chk", None))

