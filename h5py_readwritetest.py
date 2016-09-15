# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:07:59 2014

@author: Dubya
"""

import h5py
import makeimg

savefilename = "D:\\temp\\test4.hdf5"

m = 50
N = 2048
noiselevel = 0.2

fhand = h5py.File( savefilename, 'a') # file handle, saves in same directory as script
test_dset = fhand.create_dataset( "testme", (N, N, m), chunks=(N, N, 1) )

for J in range(0,m):
    print "J = " + str(J)
    test_dset[:,:,J] = makeimg.bars(N,noiselevel)
    
# Now let's try reopening the file before we close it and see what happens
    
fhand.close()

fhand = h5py.File( savefilename, 'a')
fhand2 = h5py.File( savefilename, 'a')