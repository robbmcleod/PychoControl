# -*- coding: utf-8 -*-
"""
Compilation script for the PtychoControl user interface
(better than batch files)


Created on Tues Feb 18 2014

@author: Robert A. McLeod
"""
import os
import PyQt4.uic as qu

my_path = os.path.dirname(__file__)

qu.compileUiDir( my_path )

