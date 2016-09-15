# PychoControl
Python 2.7

Requires the pyDM interface to be installed in order to record data from the Gatan camera.  Alternatively, the FEI interface
may be used, by writing and reading temporary TIFF files to disk, but there is a significant overhead from that.  

The PersistDirectAligns GUI can be used to change alignments within the microscope that the FEI interface does not allow
the user to change, for example objective stigmation in diffraction mode.  It may be invoked as:

python PDA0.2.py

The PychoControl GUI is scripted instrument control designed for recording Ptychography diffraction patterns. Realistically 
it's primary use for microscopists is as a coding example, for FEI microscopes.

PDA requires Qwt for some of the GUI elements: qwt.sourceforge.net


