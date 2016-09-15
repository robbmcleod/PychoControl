# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:07:54 2014

@author: RM238934

Python utilities for image processing by Robert A. McLeod
robbmcleod@gmail.com

"""
# REQUIRED LIBRARIES
import numpy as np
import itertools

# Try to override np.fft with pyfftw for 3x code speed-up 
# (***ALWAYS IMPORT RAMUTIL AFTER NUMPY***)
#try:
#    import pyfftw
#
#    fhand = open( "pyFFTW_wisdom.txt", 'r' )
#    oldWisdom = fhand.read()
#    pyfftw.import_wisdom( oldWisdom )
#    fhand.close()
#    # if the import succeeded, override np.fft routines
#    print( "Overriding numpy.fft with pyfftw" )
#    np.fft.fft =  pyfftw.interfaces.numpy_fft.fft
#    np.fft.ifft = pyfftw.interfaces.numpy_fft.ifft
#    np.fft.fft2 =  pyfftw.interfaces.numpy_fft.fft2
#    np.fft.ifft2 = pyfftw.interfaces.numpy_fft.ifft2
#    np.fft.fftn =  pyfftw.interfaces.numpy_fft.fftn
#    np.fft.ifftn = pyfftw.interfaces.numpy_fft.ifftn
#    np.fft.fftshift = pyfftw.interfaces.numpy_fft.fftshift
#    np.fft.ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
#    # Turn on the cache for optimum performance
#    pyfftw.interfaces.cache.enable()
#    pyfftw.interfaces.cache.set_keepalive_time( 60.0 )
#except:
#    print( "RAMutil did not find pyFFTW package: get it at https://pypi.python.org/pypi/pyFFTW" )

from scipy.optimize import curve_fit 
from scipy.stats.stats import linregress
try: 
    from skimage.filters import threshold_isodata
except ImportError:
    print( "RAMutil could not find a sufficiently recent skikit-image." )
from skimage.transform import rescale, rotate
import scipy.ndimage
from scipy.ndimage.filters import convolve, uniform_filter, gaussian_filter, median_filter
import time

import matplotlib.pyplot as plt

# DEBUG LIBRARIES
# from time import time
# import DM3lib as dm3
# import numba

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

def binWrite2( filename, inimage ):
    # Header is 8-bytes, four uint16s  
    # (ncol, nrow, bytesperpixel, typeID)
    # typeID = 1 is int32
    # typeID = 2 is float32
    # typeID = 3 is complex64 (float32*2)
    # Apparently everything must be float32 or complex64 (according to Jim)

    filehandle = open( filename, 'wb' )
    [nrow,ncol] = inimage.shape
    if( inimage.dtype == 'complex64' or inimage.dtype == 'complex128' ):
        bytesperpixel = 4
        typeID = 3
        fileheader = np.array( [ncol, nrow, bytesperpixel, typeID], dtype='uint16' )
        filehandle.write( fileheader.tostring() )
        filehandle.write( inimage.astype('complex64').tostring() )
        
    else: # Convert everything else to floating point
        bytesperpixel = 4
        typeID = 2
        fileheader = np.array( [ncol, nrow, bytesperpixel, typeID], dtype='uint16' )
        filehandle.write( fileheader.tostring() )
        filehandle.write( inimage.astype('float32').tostring() )
        
    filehandle.close()
    
def binRead2( filename, mode='zuo' ):
    # mode = 'zuo' opens as if they are DM2 with the ultra-short header
    # mode = 'koch' uses Christoph T. Koch's longer header that has pixel size and such
    try:
        filehandle = open( filename, 'rb' )
    except:
        print( "binRead2 open of file " + str(filename) + "failed" )
        return
        
    if mode == 'zuo':
        header = np.fromstring( filehandle.read(8), dtype='uint16' )
        print( "Ncol: " + str(header[0]) + ", Nrow: " + str(header[1]) + ", typeID: " + str(header[3]) )
        if header[3] == 2:
            print( "Reading float data from binimg file" )
            data = np.fromfile( filehandle, dtype='float32' )
        elif header[3] == 3:
            print( "Reading complex data from binimg file" )
            data = np.fromfile( filehandle, dtype='complex64' )
            
        filehandle.close()
        return np.reshape( data, [header[0], header[1]] )
    elif mode =='koch':
        # header = [headersize(bytes) paramSize commentSize Ny Nx complFlag doubleFlag dataSize version]
        header = np.fromfile( filehandle, count=8, dtype='int32' )
        Nx = header[4]
        Ny = header[3]
        # pix_info is pixelsize in [t, dx, dy] (or is it [t,dy,dx]?)
        pix_info = np.fromfile( filehandle, count=3, dtype='float64' )

        comment_size = header[2]
        comment = np.fromfile( filehandle, count=comment_size, dtype='|S1' )
        print( comment )
        
        # The dtype is stored in a massively over-complicated fashion
        integerFlag = 0
        complexFlag = header[5]
        doubleFlag = header[6]
        
        dtypeflag = np.bitwise_and( integerFlag + doubleFlag*4 + complexFlag*2, 7)
        print( "dtypeflag = " + str(dtypeflag) )
        if dtypeflag == 0 :
            dtype = 'float32'
        elif dtypeflag == 4:
            dtype = 'float64'
        elif dtypeflag == 2:
            dtype = 'complex64'
        elif dtypeflag == 6:
            dtype = 'complex128'
        elif dtypeflag == 1:
            dtype = 'int16'
        elif dtypeflag == 5:
            dtype = 'int32' 
        data = np.fromfile( filehandle, count=Nx*Ny, dtype=dtype )
        return np.reshape( data, [Ny, Nx] ), pix_info
    else:
        print( "Unknown bin .img file format" )
        
    

def ewavelength( V_accel = 3.0E5 ):
    """
    Return wavelength in meters for a relativistic electron.
    """
    e = np.double(1.60217646E-19)
    c = np.double(2.99792458E8)
    m = np.double(9.10938188E-31)
    h = np.double(6.626068E-34)
    
    E_o = m * c**2.0
    E = V_accel * e
    
    return  h * c / np.sqrt( 2.0 * E * E_o + E**2.0 )
    
def C_E( V_accel = 2.0E5 ):
    e = 1.60217646E-19
    c = 2.997792458E8
    m = 9.10938188E-31
    
    E_o = m*c**2
    E = V_accel * e
    return 2 * np.pi * e / ewavelength( V_accel ) * (E_o + E) / (E*(2*E_o + E))
    
def convertPSangtoPSreal( ps_ang, wavelen, N ):
    """Convert an angular pixel size to an angle, SI units (m and rad)
    Implicitely using small angle approximation sin(alpha) = alpha """
    return wavelen / (N * ps_ang)
    # Not that the reverse operation uses the exact same function, just with the 
    # substitution of ps_ang -> ps and vice versa
        
# def convertPSrealtoPSang( ps, wavelen, N ):
#     # Convert a real pixel size to an angle, SI units (m and rad)
#     return wavelen / (N * ps)
def diff3( x, axis=0 ):
    
    if x.shape[axis] < 3:
        print( "Error: diff5 needs a minimum matrix size of 3" )
        return
    # Apply global roll
    dx = ( np.roll(x,-1,axis=axis) - np.roll(x,1,axis=axis)  ) 
    # Apply end-points 
    if( axis == 0 ):
        dx[0,...] = (-3.0*x[0,...] + 4.0*x[1,...] - x[2,...]  )/2.0;
        dx[-1,...] = (-3.0*x[-1,...] +4.0*x[-2,...] -x[-3,...] ) /2.0;
    elif( axis == 1 ):
        dx[:,0,...] = (-3.0*x[:,0,...] + 4.0*x[:,1,...] - x[:,2,...]  )/2.0;
        dx[:,-1,...] = (-3.0*x[:,-1,...] +4.0*x[:,-2,...] -x[:,-3,...] ) /2.0;
    elif( axis == 2 ):
        dx[:,:,0,...] = (-3.0*x[:,:,0,...] + 4.0*x[:,:,1,...] - x[:,:,2,...]  )/2.0;
        dx[:,:,-1,...] = (-3.0*x[:,:,-1,...] +4.0*x[:,:,-2,...] -x[:,:,-3,...] ) /2.0;
    return dx
    
def diff5( x, axis=0 ):
    
    if x.shape[axis] < 5:
        print( "Error: diff5 needs a minimum matrix size of 5" )
        return
        
    # Apply global roll
    dx = ( -np.roll(x,-2,axis=axis) + 8.0*np.roll(x,-1,axis=axis) + 
        -8.0*np.roll(x,1,axis=axis) + np.roll(x,2,axis=axis) ) / 12.0
    # Apply end-points (alternative is to just pad it)
    # Really would like this to work on 3-d matrices directly too!
    if( axis == 0 ):
        dx[0,...] = (-25.0*x[0,...] +48.0*x[1,...] -36.0*x[2,:] +16.0*x[3,...] -3.0*x[4,...] )/12.0;
        dx[1,...] = (-25.0*x[1,...] +48.0*x[2,...] -36.0*x[3,:] +16.0*x[4,...] -3.0*x[5,...] )/12.0;
        dx[-1,...] = (-25.0*x[-1,...] +48.0*x[-2,...] -36.0*x[-3,:] +16.0*x[-5,...] -3.0*x[-6,...] )/12.0;
        dx[-2,...] = (-25.0*x[-2,...] +48.0*x[-3,...] -36.0*x[-4,:] +16.0*x[-6,...] -3.0*x[-7,...] )/12.0;
    elif( axis == 1 ):
        dx[:,0,...] = (-25.0*x[:,0,...] +48.0*x[:,1,...] -36.0*x[:,2,...] +16.0*x[:,3,...] -3.0*x[:,4,...] )/12.0;
        dx[:,1,...] = (-25.0*x[:,1,...] +48.0*x[:,2,...] -36.0*x[:,3,...] +16.0*x[:,4,...] -3.0*x[:,5,...] )/12.0;
        dx[:,-1,...] = (-25.0*x[:,-1,...] +48.0*x[:,-2,...] -36.0*x[:,-3,...] +16.0*x[:,-5,...] -3.0*x[:,-6,...] )/12.0;
        dx[:,-2,...] = (-25.0*x[:,-2,...] +48.0*x[:,-3,...] -36.0*x[:,-4,...] +16.0*x[:,-6,...] -3.0*x[:,-7,...] )/12.0;
    elif( axis == 2 ):
        dx[:,:,0,...] = (-25.0*x[:,:,0,...] +48.0*x[:,:,1,...] -36.0*x[:,:,2,...] +16.0*x[:,:,3,...] -3.0*x[:,:,4,...] )/12.0;
        dx[:,:,1,...] = (-25.0*x[:,:,1,...] +48.0*x[:,:,2,...] -36.0*x[:,:,3,...] +16.0*x[:,:,4,...] -3.0*x[:,:,5,...] )/12.0;
        dx[:,:,-1,...] = (-25.0*x[:,:,-1,...] +48.0*x[:,:,-2,...] -36.0*x[:,:,-3,...] +16.0*x[:,:,-5,...] -3.0*x[:,:,-6,...] )/12.0;
        dx[:,:,-2,...] = (-25.0*x[:,:,-2,...] +48.0*x[:,:,-3,...] -36.0*x[:,:,-4,...] +16.0*x[:,:,-6,...] -3.0*x[:,:,-7,...] )/12.0;
    return dx
    
        
        
def bragg_angle( atom = 'Si', V_accel = 2.0e5, hkl = [1,1,1] ):
    
    if atom is 'Au':
        a = 407.82e-12
    elif atom is 'Si':
        a = 543.09e-12
        
        
    wavelen = ewavelength( V_accel )
    hkl = np.asarray(hkl)
    
    twothetabragg = 2 * np.arcsin( 0.5 * wavelen / a * np.sqrt( np.sum(hkl**2)))
    dspace = a / np.sqrt( np.sum(hkl**2))
    return twothetabragg, dspace
    


    
def normalized(a):
    """RAM: in Pythong in-line calculations are faster
    RAM: warning, use only on float or double """
    if np.issubdtype( a.dtype, np.integer ):
        a = a.astype( 'float' )
    amin = a.min()
    arange = (a.max() - amin)
    a -= amin
    a /= arange
    return a
    
def stdfilter( image, radius):
    """
    Standard deviation filter, useful for establishing noise levels in a flat-field image
    """
    padimage = np.pad( image, radius, mode='symmetric' )
    c1 = uniform_filter( padimage, radius*2, mode='constant', origin=-radius)
    c2 = uniform_filter( padimage*padimage, radius*2, mode='constant', origin=-radius)
    # Some issue here, if c2 < c1*c1, you take sqrt of a negative number
    return ( np.sqrt(np.abs(c2 - c1*c1)) )[:image.shape[0],:image.shape[1]]
    
def fit( x, y, funchandle='gauss1', estimates=None ):
    """ Returns: fitstruct,  fitY, Rbest """
    
    if funchandle == 'gauss1':
        def fitfunc( x, a1, b1, c1 ):
            return a1 * np.exp( -( (x-b1)/ c1)**2 )
        # Really arbitrary c1 estimate at basically 25 pixels..
        if estimates is None:
            estimates = np.array( [np.max(y), x[np.argmax(y)], 25.0*(x[1]-x[0]) ] )
        
    elif funchandle == 'poly1':
        def fitfunc( x, a1, b1 ):
            return a1 * x + b1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [slope, intercept]
    elif funchandle == 'poly2':
        def fitfunc( x, a1, b1, c1 ):
            return a1 * x **2.0 + b1 *x + c1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, slope, intercept]
    elif funchandle == 'poly3':
        def fitfunc( x, a1, b1, c1, d1 ):
            return a1 * x **3.0 + b1 *x**2.0 + c1*x + d1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, 0.0, slope, intercept]
    elif funchandle == 'poly5':
        def fitfunc( x, a1, b1, c1, d1, e1, f1 ):
            return a1 * x **5.0 + b1 *x**4.0 + c1*x**3.0 + d1*x**2.0 + e1*x + f1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, 0.0, 0.0, 0.0, slope, intercept]
    elif funchandle == 'abs1':
        def fitfunc( x, a1 ):
            return a1 * np.abs( x )
        if estimates is None:
            estimates = np.array( [ (np.max(y)-np.min(y))/(np.max(x)-np.min(x))])
    elif funchandle == 'exp':
        def fitfunc( x, a1, c1 ):
            return a1 * np.exp( c1*x )
        if estimates is None:
            estimates = np.array( [1.0, -1.0] )
    elif funchandle == 'expc':
        def fitfunc( x, a1, c1, d1 ):
            return a1 * np.exp( c1*x ) + d1
        if estimates is None:
            estimates = np.array( [1.0, -1.0, 1.0] )
    elif funchandle == 'power1':
        def fitfunc( x, a1, b1 ):
            return a1*(x**b1)
        if estimates is None:
            estimates = np.array( [1.0, -2.0] )   
    elif funchandle == 'power2':
        def fitfunc( x, a1, b1, c1 ):
            return a1*(x**b1) + c1
        if estimates is None:
            estimates = np.array( [1.0, -2.0, 1.0] )    
    else:
        fitfunc = funchandle
        
    try:
        fitstruct, pcov = curve_fit( fitfunc, x, y, p0=estimates )
        perr = np.sqrt(np.diag(pcov))
        print( "Fitting completed with perr = " + str(perr) )
        fitY = fitfunc( x, *fitstruct )
        goodstruct = linregress( x, fitfunc( x, *fitstruct ) )
        Rbest = goodstruct[2]
    except RuntimeError:
        print( "RAM: Curve fitting failed")
        return
    return fitstruct,  fitY, Rbest

def imHist(imdata, bins_=256):
    '''Compute image histogram.
        [histIntensity, histX] = imHist( imageData, bins_=256 )
    '''
    im_values =  np.ravel(imdata)
    hh, bins_ = np.histogram( im_values, bins=bins_ )
    # check histogram format
    if len(bins_)==len(hh):
        pass
    else:
        bins_ = bins_[:-1]    # 'bins' == bin_edges
        
    return hh, bins_

def histClim( imdata, cutoff = 0.01, bins_ = 512 ):
    '''Compute display range based on a confidence interval-style, from a histogram
    (i.e. ignore the 'cutoff' proportion lowest/highest value pixels)'''
    
    if( cutoff <= 0.0 ):
        return imdata.min(), imdata.max()
    # compute image histogram
    hh, bins_ = imHist(imdata, bins_)
    hh = hh.astype( 'float' )

    # number of pixels
    Npx = np.sum(hh)
    hh_csum = np.cumsum( hh )
    
    # Find indices where hh_csum is < and > Npx*cutoff
    try:
        i_forward = np.argwhere( hh_csum < Npx*(1.0 - cutoff) )[-1][0]
        i_backward = np.argwhere( hh_csum > Npx*cutoff )[0][0]
    except IndexError:
        print( "histClim failed, returning min and max" )
        return np.array( [np.min(imdata), np.max(imdata)] )
    
    clim =  np.array( [bins_[i_backward], bins_[i_forward]] )
    if clim[0] > clim[1]:
        clim = np.array( [clim[1], clim[0]] )
    return clim

def plotHistogram( imdata, cutoff = 0.001 ):
    bins = np.sqrt( imdata.shape[0] * imdata.shape[1])
    histLim = histClim( imdata, cutoff = cutoff, bins_ = bins )
        
    [histData, histCounts] = np.histogram( imdata, bins = bins/4.0, range=histLim )
    histCounts = histCounts[:-1]
    
    plt.figure()
    plt.plot( histCounts, histData )
    plt.xlabel( 'Image counts' )
    plt.ylabel( 'Histogram counts' )
    plt.title( "Max at: " + str( histCounts[np.argmax(histData)] )  )
    
def histLogClim( imdata, cutoff = 0.005, bins_ = 512):
    # Just a wrapper for giving reasonable clim values for diffraction patterns
    clim_log10 = histClim( np.log10(imdata), cutoff=0.01, bins_ = 512 )
    return clim_log10 ** 10.0
    
def imScaleAndCrop( im, scale, newsize=[0,0], mode='bilinear' ):
    """
    This is an interface to interp2_??? function to resize an image 'im' and 
    then crop it or zero-pad it.  Used for resizing apertures and the like.
    """

    scale = np.asarray( scale )
    try:
        scale[1]
    except IndexError:
        scale = np.array( [scale, scale] )
        
    imsize = np.asarray( im.shape )
    if newsize[0] == 0 and newsize[1] == 0:
        newsize = np.round( imsize * scale )
        
    # Ok now with cropping
    cropRatio = newsize / (scale*imsize )
    if cropRatio[0] < 1.0:
        Mrange = np.linspace( imsize[0]/2.0 - cropRatio[0]/2.0*imsize[0], imsize[0]/2.0 + cropRatio[0]/2.0*imsize[0], newsize[0] )
    elif cropRatio[0] > 1.0:
        print( "TODO: imageShiftAndCrop cannot handle zero-padding yet" )
        pass
    else: # same size
        Mrange = np.arange( 0.0, imsize[0], 1.0/scale[0] )
    
    if cropRatio[1] < 1.0:
        Nrange = np.linspace( imsize[1]/2.0 - cropRatio[1]/2.0*imsize[1], imsize[1]/2.0 + cropRatio[1]/2.0*imsize[1], newsize[1] )
    elif cropRatio[1] > 1.0:
        print( "TODO: imageShiftAndCrop cannot handle zero-padding yet" )
        pass
    else: # same size
        Nrange = np.arange( 0.0, imsize[1], 1.0/scale[1] )    
        
    [xmesh, ymesh] = np.meshgrid( Nrange, Mrange )
    if mode == 'bilinear':
        return interp2_bilinear( im, xmesh, ymesh )
    if mode == 'nearest':
        return interp2_nn( im, xmesh, ymesh )   
    
    
def interp2_bilinear(im, x, y):
    """
    Ultra-fast interpolation routine for 2-D images.  x and y are meshes.  The 
    coordinates of the image are assumed to be 0,shape[0], 0,shape[1]
    
    BUG: This is sometimes skipping the last row and column
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # RAM: center this cliping with a roll?
    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id
    
def interp2_nn( im, x, y ):
    """
    Fast nearest neighbour interpolation, used for more advaced (filtered)
    methods such as Lanczos filtering.  x and y are meshes.  The coordinates of 
    the image are assumed to be 0,shape[0], 0,shape[1]
    """
    # We use floor instead of round because otherwise we end up with a +(0.5,0.5) pixel shift
    px = np.floor(x).astype(int)
    py = np.floor(y).astype(int)
    # Clip checking, could be more efficient because px and py are sorted...
    px = np.clip( px, 0, im.shape[1]-1 )
    py = np.clip( py, 0, im.shape[0]-1 )
    
    return im[py,px]



    
def imageShiftAndCrop( mage, shiftby, pad_value = 0.0 ):
    """ imageShiftAndCrop( mage, shiftby )
    This is a relative shift, integer pixel only, pads with zeros to cropped edges
    
    mage = input image
    shiftby = [y,x] pixel shifts    
    """
    
    # Actually best approach is probably to roll and then zero out the parts we don't want
    # The pad function is expensive in comparison

    shiftby = np.array( shiftby, dtype='int' )
    # Shift X
    if(shiftby[1] < 0 ):
        mage = np.roll( mage, shiftby[1], axis=1 )
        mage[:, shiftby[1]+mage.shape[1]:] = pad_value
    elif shiftby[1] == 0:
        pass
    else: # positive shift
        mage = np.roll( mage, shiftby[1], axis=1 )
        mage[:, :shiftby[1]] = pad_value
    # Shift Y
    if( shiftby[0] < 0 ):
        mage = np.roll( mage, shiftby[0], axis=0 )
        mage[shiftby[0]+mage.shape[0]:,:] = pad_value
    elif shiftby[0] == 0:
        pass
    else:  # positive shift
        mage = np.roll( mage, shiftby[0], axis=0 )
        mage[:shiftby[0],:] = pad_value
    return mage

# Incorporate some static vars for the meshes?
# It's fairly trivial compared to the convolve cost, but if we moved the subPixShift
# outside it's possible.
# Best performance improvement would likely be to put it as a member function in
# ImageRegistrator so that it can work on data in-place.
def lanczosSubPixShift( imageIn, subPixShift, kernelShape=3, lobes=None ):
    """ lanczosSubPixShift( imageIn, subPixShift, kernelShape=3, lobes=None )
        imageIn = input 2D numpy array
        subPixShift = [y,x] shift, recommened not to exceed 1.0, should be float
        
    Random values of kernelShape and lobes gives poor performance.  Generally the 
    lobes has to increase with the kernelShape or you'll get a lowpass filter.
    
    Generally lobes = (kernelShape+1)/2 
    
    kernelShape=3 and lobes=2 is a lanczos2 kernel, it has almost no-lowpass character
    kernelShape=5 and lobes=3 is a lanczos3 kernel, it's the typical choice
    Anything with lobes=1 is a low-pass filter, but next to no ringing artifacts
    """
    
    kernelShape = np.array( [kernelShape], dtype='int' )
    if kernelShape.ndim == 1: # make it 2-D
        kernelShape = np.array( [kernelShape[0], kernelShape[0]], dtype='int' )
        
    if lobes is None:
        lobes = (kernelShape[0]+1)/2
    
    x_range = np.arange(-kernelShape[1]/2,kernelShape[1]/2)+1.0-subPixShift[1]
    x_range = ( 2.0 / kernelShape[1] ) * x_range 
    y_range = np.arange(-kernelShape[1]/2,kernelShape[0]/2)+1.0-subPixShift[0]
    y_range = ( 2.0 /kernelShape[0] ) * y_range
    [xmesh,ymesh] = np.meshgrid( x_range, y_range )
    # xmesh and ymesh should have range [-1,1] for the Lanczos window to be proper
    
    
    # Let's try making the filter rectangular instead...
    lanczos_filt = np.sinc(xmesh * lobes) * np.sinc(xmesh) * np.sinc(ymesh * lobes) * np.sinc(ymesh)
    
    lanczos_filt = lanczos_filt / np.sum(lanczos_filt) # Normalize filter output

    imageOut = convolve( imageIn, lanczos_filt, mode='reflect' )
    return imageOut
    
# Can I write a version of this that only does a subpixel shift?
def lanczosResample( imageIn, newShape, subPixShift=[0.0,0.0], lanczos_lobes = 1.5 ):
    """TO DO (maybe): fix this so it uses imageShiftAndCrop style??? """
    # TODO: speed up :  http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter

    scaleFact = np.asarray(newShape).astype('float') / np.asarray( imageIn.shape ).astype('float')
    # So we need the mesh to scale up if we are oversampling massively.  
    # So minimum size is [5,5], maximum size is scaleFact*lanczos_lobes
    gridShape = np.clip( scaleFact*lanczos_lobes, 5, np.Inf )
    
    [xmesh,ymesh] = np.meshgrid( (np.arange(-gridShape[1],gridShape[1]+1)-subPixShift[1])/scaleFact[1],
        (np.arange(-gridShape[0],gridShape[0]+1)-subPixShift[0])/scaleFact[0] )
    rmesh = np.sqrt( xmesh*xmesh + ymesh*ymesh )
    # from plotting import ims
    
    # Works okay for slightly rectangular images.  Really strong aspect ratios 
    # cause problems like what one sees in bicubic resampling, however.
    lanczos_filt = np.sinc(rmesh)*np.sinc(rmesh/lanczos_lobes)
    lanczos_filt = lanczos_filt / np.sum(lanczos_filt) # Normalize filter output

    # skimage has an amazing requirement for floats to be in [-1,1]
    # So I wrote my own nearest-neighbour resampling code
    if np.any( np.array(imageIn.shape) != np.array(newShape) ):
        # This step is not needed if the image size doesn't change
        [xsample,ysample] = np.meshgrid( np.arange( 0,newShape[1] )/scaleFact[1], np.arange( 0,newShape[0] )/scaleFact[0] )
        imageOut = interp2_nn( imageIn, xsample, ysample )
        imageOut = convolve( imageOut, lanczos_filt, mode='reflect' )
    else:
        imageOut = convolve( imageIn, lanczos_filt, mode='reflect' )

    return imageOut
    
def magickernel( imageIn, k=1, direction='down' ):
    """ 
    magickernel( imageIn, k=1, direction='down' )
        k = number of binning operations, so k = 3 bins by 8 x 8 
    Implementation of the magickernel for power of 2 image resampling.  Generally 
    should be used to get two images 'close' in size before using a more aggressive resampling 
    method like bilinear.  
    
    direction is either 'up' (make image 2x bigger) or 'down' (make image 2x smaller)
    k is the number of iterations to apply it.
    """
    
    if k > 1:
        imageIn = magickernel( imageIn, k=k-1, direction=direction )
        
    if direction == 'up':
        h = np.array( [[0.25, 0.75, 0.75, 0.25]] )
        h = h* np.transpose(h)
        
        imageOut = np.zeros( [ 2*imageIn.shape[0], 2*imageIn.shape[1] ] )
        # Slice the input image interlaced into the larger output image
        imageOut[1::2,1::2] = imageIn
        # Apply the magic kernel
        imageOut = scipy.ndimage.convolve( imageOut, h )
        
    elif direction == 'down':
        imageIn = np.pad( imageIn, [1,1], 'reflect' )
        
        h = 0.5*np.array( [[0.25, 0.75, 0.75, 0.25]] )
        h = h* np.transpose(h)
        # This is computationally a little expensive, we are only using one in four values afterward
        imageOut = scipy.ndimage.convolve( imageIn, h)
        # Slicing is (start:stop:step)
        imageOut = imageOut[0:-2:2,0:-2:2]
    else:
        return
    return imageOut
    
def squarekernel( imageIn, k=1, direction='down' ):
    """ 
    squarekernel( imageIn, k=1, direction='down' )
        k = number of binning operations, so k = 3 bins by 8 x 8 
    Implementation of a square kernel for power of 2 image resampling, i.e. rebinning
    
    direction is either 'up' (make image 2x bigger) or 'down' (make image 2x smaller)
    k is the number of iterations to apply it.
    """
    
    if k > 3:
        # We can do this for probably bin-factors of 2,4, and 8?
        imageIn = squarekernel( imageIn, k=(k-1), direction=direction )
    
    if k == 1:
        h = np.array( [[1.0, 1.0]] )
    elif k == 2:
        h = np.array( [[1.0,1.0,1.0,1.0]] )
    elif k == 3:
        h = np.array( [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]] )
    h = h * np.transpose(h)    
    
    if direction == 'up':
        imageOut = np.zeros( [ 2*imageIn.shape[0], 2*imageIn.shape[1] ] )
        # Slice the input image interlaced into the larger output image
        imageOut[1::2,1::2] = imageIn
        # Apply the magic kernel
        imageOut = scipy.ndimage.convolve( imageOut, h )
        
    elif direction == 'down':
        # This is computationally a little expensive, we are only using one in four values afterward
        imageOut = scipy.ndimage.convolve( imageIn, h )
        # Slicing is (start:stop:step)
        imageOut = imageOut[1:-2:2,1:-2:2]
    else:
        return
    return imageOut
    
def apodization( name = 'butter.32', size = [2048,2048], radius=[2048.0,2048.0] ):
    """
    Provides a 2-D filter or apodization window for Fourier filtering or image clamping.
    
    Valid names are (or will be): 
        'hann'
        'hamming'
        'butter.X' where X is the order of the Lorentzian
        'gauss_trunc' - truncated gaussian
        'gauss' - regular gaussian
        'lanczos' - dampened sinc
    NOTE: There are windows in scipy.signal for 1D-filtering...
    """
    # Make meshes
    # Warning: this assumes the input size is a power of 2
    radius = np.asarray( radius, dtype='float' )
    # DEBUG: Doesn't work right for odd numbers
    [xmesh,ymesh] = np.meshgrid( np.arange(-size[1]/2,size[1]/2), np.arange(-size[0]/2,size[0]/2) )
    r2mesh = xmesh*xmesh/( np.double(radius[0])**2 ) + ymesh*ymesh/( np.double(radius[1])**2 )
    
    try:
        [name, order] = name.lower().split('.')
        order = np.double(order)
    except ValueError:
        order = 1
        
    if name == 'butter':
        window =  np.sqrt( 1.0 / (1.0 + r2mesh**order ) )
    elif name == 'butter_square':
        window = np.sqrt( 1.0 / (1.0 + (xmesh/radius[1])**order))*np.sqrt(1.0 / (1.0 + (ymesh/radius[0])**order) )
    elif name == 'hann':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * 0.5 * ( 1.0 + np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hann_square':
        window = ( (0.5 + 0.5*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.5 + 0.5*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'hamming':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') *  ( 0.54 + 0.46*np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hamming_square':
        window = ( (0.54 + 0.46*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.54 + 0.46*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'gauss' or name == 'gaussian':
        window = np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'gauss_trunc':
        cropwin = ((0.5*xmesh/radius[1])**2.0 + (0.5*ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'lanczos':
        print( "TODO: Implement Lanczos window" )
        return
    else:
        print( "Error: unknown filter name passed into apodization" )
        return
    return window
    

def lowpass2( mage, name = 'butter.32', radius=[512.0,512.0], padding=[0,0] ):
    """
    Function takes an image mage, pads it out, and then applies a low-pass filter 
    in Fourier-space before removing the padding once again.
    """
    mshape = mage.shape
    magebig = np.pad( mage, (padding), mode='symmetric' )
    lpfilter = np.fft.fftshift( apodization( name=name, size=magebig.shape, radius=radius ) )

    # Not really sure how to retain the sign of the image properly?
    # if ( mage < 0.0 ).any():
    #     print( "WARNING: lowpass2 has trouble with negative data." )
    # CHANGED NP.ABS TO NP.REAL to try and mitigate impact on negative numbers in image...
    magebig = np.real( np.fft.ifft2( lpfilter * np.fft.fft2( magebig ) ) )
    magebig = np.roll( np.roll( magebig, -padding[1], axis=1 ), -padding[0], axis=0 )
    magebig = magebig[ 0:mshape[0], 0:mshape[1] ]
    return magebig
    
# Add rotmean attributes so that we don't repeatedly compute the overhead if the 
# image size is the same: http://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
@static_var( "rfloor", 0 )
@static_var( "remain", 0 )
@static_var( "remain_n", 0 )
@static_var( "weights", 0 )
@static_var( "raxis", 0 )
@static_var( "prevN", 0 )
@static_var( "prevM", 0 )
@static_var( "weights", 0 )
@static_var( "raxis", 0 )
def rotmean( mage ):
    """
    Computes the rotational mean about the center of the image.  Generally used 
    on the magnitude of Fourier transforms. Uses static variables that accelerates 
    the precomputation of the meshes if you call it repeatedly on the same 
    dimension arrays. 
    
    NOTE: returns both rmean, raxis so you must handle the raxis part.
    
    Mage should be a power of two.  If it's not, it's padded automatically
    """
    if np.mod( mage.shape[1],2 ) == 1 and np.mod( mage.shape[0],2) == 1:
        mage = np.pad( mage, ((0,1),(0,1)), 'edge' )
    elif np.mod( mage.shape[1],2 ) == 1:
        mage = np.pad( mage, ((0,0),(0,1)), 'edge' )
    elif np.mod( mage.shape[0],2 ) == 1:
        mage = np.pad( mage, ((0,1),(0,0)), 'edge' )    
        
    N = int( np.floor( mage.shape[1]/2.0 ) )
    M = int( np.floor( mage.shape[0]/2.0 ) )
    
    if N != rotmean.prevN or M != rotmean.prevM:
        # Initialize everything
        rotmean.prevN = N
        rotmean.prevM = M
        
        rmax = np.ceil( np.sqrt( N**2 + M**2 ) ) + 1
        [xmesh, ymesh] = np.meshgrid( np.arange(-N, N), np.arange(-M, M) )
        rmesh = np.sqrt( xmesh**2 + ymesh**2 )
        rotmean.rfloor = np.floor( rmesh )
        
        rotmean.remain = rmesh - rotmean.rfloor
        # Make rfloor into an index look-up table
        rotmean.rfloor = rotmean.rfloor.astype(np.int).ravel()
        
        # Ravel
        rotmean.remain = rotmean.remain.ravel()
        rotmean.remain_n = 1.0 - rotmean.remain
        
        rotmean.weights = np.zeros( [rmax] )
        weights_n = np.zeros( [rmax] )
        
        weights_n[rotmean.rfloor] += rotmean.remain_n
        rotmean.weights[ (rotmean.rfloor+1) ] = rotmean.remain
        rotmean.weights += weights_n
        rotmean.raxis = np.arange(0,rotmean.weights.size)
    else:
        # Same size image as previous time
        # Excellent now only 150 ms in here for 2k x 2k...
        # Rotmean_old was 430 ms on the desktop
        pass
    
    # I can flatten remain and mage
    mage = mage.ravel()
    mage_p = mage * rotmean.remain
    mage_n = mage * rotmean.remain_n

    rmean = np.zeros( np.size(rotmean.weights) )
    rmean_n = np.zeros( np.size(rotmean.weights) )

    # Find lower ("negative") remainders
    rmean_n[rotmean.rfloor] += mage_n
    
    # Add one to indexing array and add positive remainders to next-neighbours in sum
    rmean[ (rotmean.rfloor+1) ] = mage_p
    
    # sum
    rmean += rmean_n
    # and normalize sum to average
    rmean /= rotmean.weights
    
    return [rmean, rotmean.raxis]


def rotmean_old( mage ):
    N = int( np.floor( mage.shape[0]/2.0 ) )
    M = int( np.floor( mage.shape[1]/2.0 ) )
    rmax = np.ceil( np.sqrt( N**2 + M**2 ) ) + 1
    
    [xmesh, ymesh] = np.meshgrid( np.arange(-N, N), np.arange(-M, M) )
    rmesh = np.sqrt( xmesh**2 + ymesh**2 )
    rfloor = np.floor( rmesh )
    
    remain = rmesh - rfloor
    # Make rfloor into an index look-up table
    rfloor = rfloor.astype(np.int)
    
    # I can flatten remain and mage
    mage = mage.ravel()
    remain = remain.ravel()
    # remain_n = np.ones( remain.shape ) - remain;
    remain_n = 1.0 - remain;
    rfloor = rfloor.ravel()
    
    # Note that we don't touch the input image until here.  Everything above could 
    # be pre-computed and held in memory somehow with globals?
    # Python 3.3. has a @functools.lru_cache but this only saves the return values
    # I could write a memoized subfunction that returns all the values for a given [N,M]?
    # or a lookup table...
    # Or make RAMUtil a singleton class...
    # Or take advantage of the fact rotmean is an object?
    # rotmean.rmesh, rotmean.rfloor
    mage_p = mage*remain
    mage_n = mage*remain_n
    
    # Somewhat better initialization time (~200 ms) but still slow...
    rmean = np.zeros( [rmax] )
    rmean_n = np.zeros( [rmax] )
    weights = np.zeros( [rmax] )
    weights_n = np.zeros( [rmax] )

    # Find lower ("negative") remainders
    rmean_n[rfloor] += mage_n
    weights_n[rfloor] += remain_n
    
    # Add one to indexing array and add positive remainders to next-neighbours in sum
    rfloor += 1
    rmean[rfloor] = mage_p
    weights[rfloor] = remain
    
    # sum
    rmean += rmean_n
    weights += weights_n
    # and normalize sum to average
    rmean /= weights
    
    # I do not understand why but array is returned backwards (in any case it's probably faster to 
    # manupulate small 1-D arrays)
    # rmean = rmean[::-1]
    raxis = np.arange(0,rmean.size)
    return [rmean, raxis]
    
def imRotate( image, theta, order = 2, doResize=False ):
    """
    Rotate an image using skimage.transform.rotate.  Basically just a wrapper to 
    deal with the fact that skimage thinks floating point images need to be between [-1.0,1.0]
    
    theta is in RADIANS
    order is order of spline interpolation.  2 and 4 works well, but order 4 takes x12 longer!
    ( About 0.4 s versus 6 s on Xeon E5-2650 @ 2 GHz )
    
    """
    imagemin = np.min(image)
    imagerange = np.max(image) - imagemin
    image -= imagemin
    image /= imagerange
    image = rotate( image, theta * 180.0 / np.pi, order = order, resize=doResize )
    image *= imagerange
    image += imagemin
    return image
    
def imRescale( image, newscale, order=1, maintainSize=False ):
    """
    Rescales an image, cropping or symmetric padding as necessary to retain the 
    same size as the original image
    """
    imageShapeOld = image.shape
    
    # TO DO: add testing for float status
    imagemin = np.min(image)
    imagerange = np.max(image) - imagemin
    image -= imagemin
    image /= imagerange
    image = rescale( image, newscale, order=order )
    image *= imagerange
    image += imagemin
    
    # This doesn't do any cropping or padding, which is what I typically need.
    diffx = image.shape[0] - imageShapeOld[0]
    diffy = image.shape[1] - imageShapeOld[1]
    if( image.shape[0] < imageShapeOld[0] ):
        # Padding
        if np.mod(diffx,2) == 0:
            padx0 = -diffx/2
            padx1 = padx0
        else:
            padx0 = np.floor(-diffx/2)+1
            padx1 = np.floor(-diffx/2)
        if np.mod(diffy,2) == 0:
            pady0 = -diffy/2
            pady1 = pady0
        else:
            pady0 = np.floor(-diffy/2)+1
            pady1 = np.floor(-diffy/2)
        
        image = np.pad( image, ((padx0,padx1),(pady0,pady1)), mode='constant' )
        
        pass
    elif( image.shape[0] > imageShapeOld[0] ):
        # Cropping
        if np.mod(diffx,2) == 0:
            cropx0 = diffx/2
            cropx1 = cropx0
        else:
            cropx0 = np.floor(diffx/2)+1
            cropx1 = np.floor(diffx/2)
        if np.mod(diffy,2) == 0:
            cropy0 = diffy/2
            cropy1 = cropy0
        else:
            cropy0 = np.floor(diffy/2)+1
            cropy1 = np.floor(diffy/2)
        
        # print( 'x0: ' + str(cropx0) + ', x1: ' + str(cropx1) + ", y0:" + str(cropy0) + ", y1:" + str(cropy1) )
        image = image[cropx0:-cropx1,cropy0:-cropy1]
        pass
    
    return image
    


def fermat_spiral(a,n):
    """
    Creates a Fermat spiral with n points distributed in a circular area with 
    diamter<= dmax. Returns the x,y coordinates of the spiral points. The average
    distance between points can be roughly estimated as 0.5*dmax/(sqrt(n/pi))
    
    RAM comment: setting it to an overlap distance would be more helpful, so that the input 
    parameter a = 0.5*dmax/(sqrt(n/pi)) was used instead?  That is,
    dmax = 2*a*sqrt(n/pi)
    
    http://en.wikipedia.org/wiki/Fermat%27s_spiral
    """
    dmax = 2*a*np.sqrt( n / np.pi )
    print( "Generating Fermat spiral with dmax: " + str(dmax) )
    c = 0.5*dmax/np.sqrt(n)
    vr,vt = [],[]
    t = 0.4
    goldenAngle = np.pi*(3-np.sqrt(5))
    while t < n:
        vr.append( c*np.sqrt(t) )
        vt.append( t*goldenAngle )
        t += 1
    vt, vr = np.array(vt), np.array(vr)
    return np.array( [vr*np.cos(vt),vr*np.sin(vt)] )
    
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def hexmesh( a, n ):
    """ This function is for generating hexagonal scan patterns, of spacing 'a' and N ranks (radii) 
    Number of hexagons as a function of rank is N = 1 + sum(6*rank) """
    rank = 0
    count = 1
    while count < n:
        rank += 1
        count += rank*6
        
    # Need to repeat each element 3 times or permutations doesn't give us all possible ones
    rankarray = np.concatenate( (range(-rank,rank+1), range(-rank,rank+1), range(-rank,rank+1)) )
    abg = list( itertools.permutations( rankarray, 3 ))
    
    # Convert to np    
    abg = np.array( abg )
    xy = np.array([ abg[:,0] + 0.5*( abg[:,1] - abg[:,2] ),  np.sqrt(0.75)*( abg[:,1] + abg[:,2] ) ] )
    xy = unique_rows( xy.transpose() )
    xy = xy.transpose()
    
    # Find uniques
    # xy = list( set( list(xy) ))
    
    radius_xy = np.sqrt( np.abs(xy[0,:])**2 + np.abs(xy[1,:])**2 )
    # 100 is just a fudge-factor to make radius dominate over angle in the ordering
    theta_xy = np.arctan2( xy[1,:], xy[0,:] ) + 100*np.pi*radius_xy
    
    index_theta = theta_xy.argsort()
    
    mesh_out = xy[:,index_theta]
    
    # Crop
    mesh_out = mesh_out[:,0:n]
    mesh_out = np.array( mesh_out )
    # Apply scaling (causes problems with uniques detection if it's too small, 
    # and applied early, Python does something wierd...)
    mesh_out *= a
    
    return mesh_out
    

def gridmesh( a, N, M ):
    """This function generates a rectangular mesh of scan coordinates of spacing 'a'
    Note that while a hexagonal mesh has a higher packing density for a circular probe,
    often displaying and managing data with gridmesh is easier.
    In Matlab style, N is number of x-points, M number of y-points """
    xvect = np.linspace( -( (N-1)*a/2), ( (N-1)*a/2), N )
    yvect = np.linspace( -( (M-1)*a/2), ( (M-1)*a/2), M )
    x_mesh, y_mesh = np.meshgrid( xvect, yvect )
    mesh_out = np.array( [np.ravel( x_mesh ), np.ravel(y_mesh) ] )
    return mesh_out
    
def rotatemesh( coords, gamma ):
    """Rotate some 2-d coordinates by the angle gamma"""
    coords_out = np.zeros( coords.shape )
    coords_out[0,:] = np.cos( coords[0,:] ) - np.sin(coords[1,:]  )
    coords_out[1,:] = np.sin( coords[0,:] ) + np.cos(coords[1,:]  )
    return coords_out
    
def probeCenter( probe ):
    """
    Probe centering by center-of-mass methodology
    """
    profprobe_x = np.sum( probe, 0, dtype=float )
    profprobe_y = np.sum( probe, 1, dtype=float )
    # Image must be background subtracted, or else we will blow up.  
    profprobe_x = normalized(profprobe_x)
    profprobe_y = normalized(profprobe_y)
    
    csprobe_x = np.cumsum(profprobe_x)
    csprobe_y = np.cumsum(profprobe_y)
    
    csprobe_x = normalized(csprobe_x)
    csprobe_y = normalized(csprobe_y)
    
    probepos_x = np.argwhere( csprobe_x >= 0.5 )[0][0]
    probepos_y = np.argwhere( csprobe_y >= 0.5 )[0][0]
    probepos = np.array( [probepos_y, probepos_x] )
    #print( str(probepos_x) + ", " + str(probepos_y) )
    
    return probepos
    
def probeCenterAndShift( probe ):
    """
    Probe centering by center-of-mass methodology
    """
    profprobe_x = np.sum( probe, 0, dtype=float )
    profprobe_y = np.sum( probe, 1, dtype=float )
    # Image must be background subtracted, or else we will blow up.  
    profprobe_x = normalized(profprobe_x)
    profprobe_y = normalized(profprobe_y)
    
    csprobe_x = np.cumsum(profprobe_x)
    csprobe_y = np.cumsum(profprobe_y)
    
    csprobe_x = normalized(csprobe_x)
    csprobe_y = normalized(csprobe_y)
    
    probepos_x = np.argwhere( csprobe_x >= 0.5 )[0][0]
    probepos_y = np.argwhere( csprobe_y >= 0.5 )[0][0]
    probepos = np.array( [probepos_y, probepos_x] )
    #print( str(probepos_x) + ", " + str(probepos_y) )
    
    M = probe.shape[0]
    N = probe.shape[1]    
    
    probe = np.roll( np.roll( probe, -(probepos_y-M/2).astype('int64'), axis=0 ), -(probepos_x-N/2).astype('int64'), axis=1 )
    return probe, probepos
    
def probeCenterAndAstig( probe ):
    """
    Probe centering by center-of-mass methodology, with width measurement for 
    rough astigmatism
    """
    profprobe_x = np.sum( probe, 0, dtype=float )
    profprobe_y = np.sum( probe, 1, dtype=float )
    # Image must be background subtracted, or else we will blow up.  
    profprobe_x = normalized(profprobe_x)
    profprobe_y = normalized(profprobe_y)
    
    csprobe_x = np.cumsum(profprobe_x)
    csprobe_y = np.cumsum(profprobe_y)
    
    csprobe_x = normalized(csprobe_x)
    csprobe_y = normalized(csprobe_y)
    
    probepos_x = np.argwhere( csprobe_x >= 0.5 )[0][0]
    probepos_y = np.argwhere( csprobe_y >= 0.5 )[0][0]
    probepos = np.array( [probepos_y, probepos_x] )
    #print( str(probepos_x) + ", " + str(probepos_y) )
    
    upslope_y = np.nonzero( profprobe_y > 0.5)[0][0]
    downslope_y = np.nonzero( profprobe_y > 0.5)[0][-1]
    upslope_x = np.nonzero( profprobe_x > 0.5)[0][0]
    downslope_x = np.nonzero( profprobe_x > 0.5)[0][-1]
    # print "up_y : " + str(upslope_y) + ", down_y : " + str(downslope_y)
    # print "up_x : " + str(upslope_x) + ", down_x : " + str(downslope_x)
    
    probewidth = np.array( [downslope_y - upslope_y, downslope_x- upslope_x])
    
    return probepos, probewidth
    
def xcorr2_fft( mage, template ):
    """Function takes in an image mage, and a template, and finds the shift between
    them by the (non-normalized)  FFT x-correlation.  """

    xc = np.abs( np.fft.ifftshift(np.fft.ifft2( np.fft.fft2(template) * np.fft.fft2(mage).conj()  )) )
    ij = np.unravel_index(np.argmax(xc), xc.shape ) # Because Python is wierd
    shift_xc = ij[::-1]
    shift_xc -= np.divide( xc.shape, 2)
    return shift_xc 
    
def probeParams( probe, doFitBack = False, doFast = False, returnRotProbe = False ):
    """
    Function accepts an image of a probe (or some other symmetric object) and 
    centers it based on the cross-sectional FWHM positions
    
        output = [probe_hwhm, probe_hwtm, position(x,y), probe_norm]
        hwhm = half-width half-maximum (in pixels)
        hwtm = half-width tenth-maximum (in pixels)
        position = original probe position (in pixels)
        probe_norm = intensity level probe was normalized by (in counts)
        
    Probe is normalized by masking it and then finding a Gaussian best-fit.  If 
    the best-fit fails to achieve a good correlation coefficient, normalization 
    is done by the mean.  The half-maxima positions are interpolated for sub-pixel 
    precision
    
    doFast = True turns off median filtering and does not center or background 
    subtract the probe
    
    """
    def gauss1(x, a, b, c):
        return a * np.exp( -((x-b)/c)**2 )
        
    # Probe centration and background removal
    [N, M] = np.shape( probe )
    if not bool(doFast): 
        profprobe_x = np.sum( probe, 0, dtype=float )
        profprobe_y = np.sum( probe, 1, dtype=float )
        # Image must be background subtracted, or else we will blow up.  
        profprobe_x = normalized(profprobe_x)
        profprobe_y = normalized(profprobe_y)
        
        csprobe_x = np.cumsum(profprobe_x)
        csprobe_y = np.cumsum(profprobe_y)
        
        csprobe_x = normalized(csprobe_x)
        csprobe_y = normalized(csprobe_y)
    
        probe_pos = np.array( [np.argwhere( csprobe_y >= 0.5 )[0][0], np.argwhere( csprobe_x >= 0.5 )[0][0]] )
        
        # Shift probe to center (to within 1 pixel)
        probe = np.roll( np.roll( probe, -(probe_pos[0]-M/2).astype('int64'), axis=0 ), -(probe_pos[1]-N/2).astype('int64'), axis=1 )
    else:
        probe_pos = np.array( [N/2,M/2] )
    # End probe centration
        
    # Median filter is quite slow (~1 s) but very often necessary
    if not bool(doFast):
        probe = normalized( median_filter(probe, size=[3,3], mode='constant' ) ) 
    else:
        probe = normalized( probe )
        
    # Now that we have a probe mask we can take a histogram of the probe, and try and fit a 
    # Gaussian to that.  If the least-squares fails, we can just take the mean, to
    # find the mean current density inside the probe.  We also need to find the mean density outside 
    # the probe so we can normalize properly...

    [probe_rotmean, _] = rotmean( probe )
    
    mask2d = probe > 0.20
    [probehist, probebins] = imHist(probe[mask2d], bins_ = 100 )
    [backhist, backbins] = imHist(probe[ np.invert(mask2d) ], bins_ = 100 )
    ### Probe   
    try:
        fit_probe, _ = curve_fit( gauss1, probebins, probehist, p0=[ probehist.max(), 1.0, 0.1] )
        good_probe = linregress( probehist, gauss1( probebins,fit_probe[0], fit_probe[1], fit_probe[2] ) )
        R_probe = good_probe[2]
        
    except RuntimeError:
        R_probe = -1.0
        fit_probe = [0.0, 0.0, 0.0]
    # Curve_fit does not have boundary limits so we have to impose them manually.
    if fit_probe[1] < 0.0:
        R_probe = -1.0
    
    if R_probe < 0.7:
        mask2d = probe > 0.4
        fit_probe[1] = probe[mask2d].mean()
        fit_probe[2] = probe[mask2d].std()
        #print "probeParam: probe intensity reverting to masked statistics, R2 = " + '{:.3f}'.format(R_probe) + ", normalized by " + '{:.3f}'.format(fit_probe[1])
    else:
        pass
        # print "probeParam: probe intensity using Gaussian best-fit, R2 = " + '{:.3f}'.format(R_probe) + ", normalized by " + '{:.3f}'.format(fit_probe[1])
        
    ### Background
    try:
        fit_back, _ = curve_fit( gauss1, backbins, backhist, p0=[ backhist.max(), 0.05, 0.05] )
        
        good_back = linregress( backhist, gauss1( backbins, fit_back[0], fit_back[1], fit_back[2] ) )
        R_back = good_back[2]
        
#        plt.figure()
#        plt.plot( backbins[:-1], backhist )
#        plt.show(block=False)
#        print( "fit_back = " + str(fit_back) + ", R_back = " + str(R_back) )
    except RuntimeError:
        R_back = -1.0
        fit_back = [0.0, 0.0, 0.0]
    # Curve_fit does not have boundary limits so we have to impose them manually.
    if fit_back[1] < 0.0:
        R_back = -1.0
    if R_back < 0.7:
        mask2d = probe < 0.05
        fit_back[1] = probe[mask2d].mean()
        fit_back[2] = probe[mask2d].std()
        #print "probeParam: background intensity reverting to masked statistics, R2 = " + '{:.3f}'.format(R_back) + ", normalized by " + '{:.3f}'.format(fit_back[1])
    else:
        pass
        #print "probeParam: background intensity using Gaussian best-fit, R2 = " + '{:.3f}'.format(R_back) + ", normalized by " + '{:.3f}'.format(fit_back[1])
    
    
    # Normalize probe with best-fits to masked histograms
    probe = (probe - fit_back[1]) / (fit_probe[1]-fit_back[1])

    # Find 1-D rotation average
    rot_probe, rot_axis = rotmean( probe )
    
    try:
        halfmaxindex = np.argwhere( rot_probe > 0.5)[-1]
        # Interpolate HWHM and HWTM
        try:
            sub_rot = rot_probe[halfmaxindex-2:halfmaxindex+3]
            # Again with interpolate requiring a flipped array
            # Try np.interp instead
            hwhm = np.interp( 0.5, sub_rot[::-1], rot_axis[halfmaxindex-2:halfmaxindex+3][::-1] )
            # hwhm = scipy.interpolate.interp1d( sub_rot[::-1], rot_axis[halfmaxindex-2:halfmaxindex+3][::-1], 'cubic' )(0.5)
        except (IndexError, ValueError):
            hwhm = halfmaxindex
    except IndexError:
        hwhm = 0.0
        
    try:
        tenthmaxindex = np.argwhere( rot_probe > 0.1)[-1]
        try:
            sub_rot = rot_probe[tenthmaxindex-2:tenthmaxindex+3]
            # I don't think scipy can do a lanczos kernel?
            # Again with interpolate requiring a flipped array
            hwtm = np.interp( 0.1, sub_rot[::-1], rot_axis[tenthmaxindex-2:tenthmaxindex+3][::-1] )
            #hwtm = scipy.interpolate.interp1d( sub_rot[::-1], rot_axis[tenthmaxindex-2:tenthmaxindex+3][::-1], 'cubic' )(0.1)
        except (IndexError, ValueError):
            hwtm = tenthmaxindex
    except IndexError:
        hwtm = 0

    hwtm = np.float( hwtm )
    hwhm = np.float( hwhm )
    
    # Given all the trouble you go to in order to get a normalized value maybe you should have a flag to 
    # return the normalized probe/rot_probe as well?
    outparams = [hwhm, hwtm, probe_pos, fit_probe[1], R_probe]
    if returnRotProbe:
        # Warning if the image is rectangular you will have some trouble here
        rot_probe = rot_probe[0:N/2]
        rot_axis = rot_axis[0:N/2]
        return outparams, rot_probe, rot_axis
    else:
        return outparams

def findPeaks2( mage, peak_rad=5.0, median_diam = 3, threshold = 0.0, edge = 1, subpix='none', showPlot=False ):
    """Return the pixel positions of the peaks, in [y,x] coords
    Note that I've change threshold to be the number to divide the (filtered) maximum value pixel by """

    subpix = subpix.lower()
    # The Gaussian filter chokes if the data is outside a [-1,1] range
    mage = normalized( np.copy(mage) )
    
    # Apply filters
    if median_diam > 0:
        mage = median_filter( mage, size = median_diam )
        
    if peak_rad > 0:
        mage = gaussian_filter( mage, peak_rad )
        
    # import plotting
    
        
    # Threshold on the filtered image (this is different order from Matlab implementation)
    if threshold == 0.0:
        threshold = threshold_isodata( mage )
        print( "Threshold level: " + str(threshold) )
        mask_peaks = mage >= threshold
        plt.figure()
        plt.imshow( mask_peaks * mage )
    else:
        mask_peaks = mage >= mage.max()/threshold
        
    # Do rolling to find local maxima
    mask_peaks *= ( mage > np.roll(mage, 1, axis=0) )
    mask_peaks *= ( mage > np.roll(mage, -1, axis=0) )
    mask_peaks *= ( mage > np.roll(mage, 1, axis=1) )
    mask_peaks *= ( mage > np.roll(mage, -1, axis=1) )
    mask_peaks *= ( mage > np.roll( np.roll(mage, 1, axis=0), 1, axis=1 ) )
    mask_peaks *= ( mage > np.roll( np.roll(mage, -1, axis=0), -1, axis=1 ) )
    mask_peaks *= ( mage > np.roll( np.roll(mage, 1, axis=0), -1, axis=1 ) )
    mask_peaks *= ( mage > np.roll( np.roll(mage, -1, axis=0), 1, axis=1 ) )
    
    # Apply edge
    mask_peaks[0:,-edge:-1] = False
    mask_peaks[0:,0:edge] = False
    mask_peaks[-edge:-1,0:] = False
    mask_peaks[0:edge,0:] = False
    
    pk_loc = np.argwhere( mask_peaks )
    # Sort pk_loc by intensity, from maximum to minimum
    
    pk_indices = np.argsort( mage.ravel()[ np.ravel_multi_index( [pk_loc[:,0],pk_loc[:,1]], mage.shape ) ] )
    # argsort is minimum to max, I need to flip
    pk_indices = pk_indices[::-1]
    pk_loc = pk_loc[pk_indices,:]
    
    if showPlot:
        # assume the user has called plt.figure() and we are just updating it
        # plt.cla()
        plt.figure()
        plt.imshow( np.log10( mage ) )
        plt.plot( pk_loc[:,1], pk_loc[:,0], 'w+', markersize=12, linewidth=2.5 )
        plt.show( block=False )
        plt.xlim( [0, mage.shape[1]] )
        plt.ylim( [0, mage.shape[0]] )
        plt.pause(0.05)
    
    print( "findPeaks2: there are no subpixel routines at present: " + subpix )
    
    return pk_loc
    pass #findPeaks2
    
def findPeaksNN( pkLoc ):
    """
    Function takes a list of peak-locations from a 2-d image, such as that outputed
    by findPeaks2 above.  It returns a list of the distances to the nearest neighbour
    peak, and the index of pkLoc where the nearest neighbour is.
    
    Useage:
        [pkNNdistance, pkNNindex] = findPeaksNN( pkLoc )
    """
    pkNNindex = np.zeros(pkLoc.shape[0])
    pkNNdistance = np.zeros(pkLoc.shape[0])
    for I in np.arange(0,pkLoc.shape[0]):
        currLoc = pkLoc[I,:]
        minDist = np.Inf
        minIndex = 0
        for J in np.arange(0,pkLoc.shape[0]):
            distanceToPk = np.sqrt( np.sum( (currLoc - pkLoc[J,:])**2.0 ) )
            if distanceToPk > 0.0:
                if distanceToPk < minDist:
                    minDist = distanceToPk
                    minIndex = J
            pass
        pass
        pkNNindex[I] = minIndex
        pkNNdistance[I] = minDist
    pass
    return pkNNdistance, pkNNindex
    
def makeMovie( imageStack, movieName, clim=None, gaussSigma=None, frameRate=5 ):
    """
    Take an image stack and save it as a bunch of JPEGs, then make a movie with 
    FFMPEG.
    """
    import os
    import skimage.io
    fex = '.tif'
    print( "RAMutil.makeMovie must be able to find FFMPEG on the system path" )
    print( "Strongly recommended to use .mp4 extension" )
    
    m = imageStack.shape[0]
    # Note that FFMPEG starts counting at 0.  
    for J in np.arange(0,m):
        mage = imageStack[J,:,:]
        if gaussSigma is not None:
            mage = gaussian_filter( mage, gaussSigma )
        if clim is not None:
            mage = np.clip( mage, clim[0], clim[1] )
        mage = (255.0*normalized(mage)).astype('uint8')
        # Can convert to colormap as follows: Image.fromarray( np.uint8( cm.ocean_r(stddesk)*255))
        try:
            skimage.io.imsave( "input_%05d"%J + fex, mage, plugin='freeimage' )
        except:
            try:
                skimage.io.imsave( "input_%05d"%J + fex, mage, plugin='tifffile' )
            except:
                skimage.io.imsave( "input_%05d"%J + fex, mage )
        pass
    time.sleep(0.5)
    
    # Remove the old movie if it's there
    try: 
        os.remove( movieName )
    except:
        pass
    
    # Make a movie with lossless H.264
    # One problem is that H.264 isn't compatible with PowerPoint.  Can use Handbrake to make it so...
    # Framerate command isn't working...
    comstring = "ffmpeg -r "+str(frameRate)+ " -f image2 -i \"input_%05d"+fex+"\" -c:v libx264 -preset veryslow -qp 0 -r "+str(frameRate)+ " "+movieName
    # comstring = "ffmpeg -r "+str(frameRate)+ " -f image2 -i \"input_%05d"+fex+"\" -c:v libx264 -preset veryslow -qp 0 -r "+str(frameRate)+ " "+moviename
    print( comstring )
    os.system( comstring )
    # Clean up
    for J in np.arange(0,m):
        os.remove( "input_%05d"%J + fex )
    pass
    
def pyFFTWPlanner( realMage, fouMage=None, effort = 'FFTW_MEASURE', cpu_cores = None, doForward = True, doReverse = True ):
    """
    Appends an FFTW plan for the given realMage to a text file stored in the same
    directory as RAMutil, which can then be loaded in the future with pyFFTWLoadWisdom.
    
    NOTE: realMage should be typecast to 'complex64' normally.
    
    Note: planning pickle files are hardware dependant, so don't copy them from one 
    machine to another.
    
    TODO: test rfft2 and irfft2 as built
    """
    import pyfftw
    import pickle
    import os.path
    from multiprocessing import cpu_count
    
    rampath = os.path.dirname(os.path.realpath(__file__))
    
    # First import whatever we already have
    try:
        fh = open( rampath + "/pyFFTW_wisdom.pkl", 'rb')
        pyfftw.import_wisdom( pickle.load( fh ) )
        fh.close()
    except:
        print( "RAMutil no previous pyFFTW wisdom found at: " + rampath + "/pyFFTW_wisdom.pkl" )
        
        # I think the fouMage array has to be smaller to do the real -> complex FFT?
    if fouMage is None:
        if realMage.dtype.name == 'float32':
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype =  'complex64'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        elif realMage.dtype.name == 'float64': 
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype = 'complex128'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        else: # Assume dtype is complexXX
            fouDtype = realMage.dtype.name
            fouMage = np.zeros( realMage.shape, dtype=fouDtype )
            
    if cpu_cores is None:
        cpu_cores = cpu_count()
    print( "Using: " + str(cpu_cores) + " cpu cores" )
    
    if bool(doForward):
        print( "Planning forward pyFFTW for shape: " + str( realMage.shape ) )
        # FFT2 = pyfftw.FFTW( realMage, fouMage, direction="FFTW_FORWARD", flags=(effort,), threads=cpu_count() )
        FFT2 = pyfftw.builders.fft2( realMage, planner_effort=effort, threads=cpu_cores )
        # FFT2 = pyfftw.builders._utils._FFTWWrapper(realMage, fouMage, direction="FFTW_FORWARD", flags=(effort,), threads=cpu_count() )
    else:
        FFT2 = None
    if bool(doReverse):
        print( "Planning reverse pyFFTW for shape: " + str( realMage.shape ) )
        # IFFT2 = pyfftw.FFTW( fouMage, realMage, direction="FFTW_BACKWARD", flags=(effort,), threads=cpu_count() )
        IFFT2 = pyfftw.builders.ifft2( fouMage, planner_effort=effort, threads=cpu_cores )
        # IFFT2 = pyfftw.builders._utils._FFTWWrapper( fouMage, realMage, direction="FFTW_BACKWARD", flags=(effort,), threads=cpu_count() )
    else: 
        IFFT2 = None

    # Setup so that we can call .execute on each one without re-copying arrays
    if FFT2 is not None and IFFT2 is not None:
        FFT2.update_arrays( FFT2.get_input_array(), IFFT2.get_input_array() )
        IFFT2.update_arrays( IFFT2.get_input_array(), FFT2.get_input_array() )
    # Something is different in the builders compared to FFTW directly. 
    # Can also repeat this for pyfftw.builders.rfft2 and .irfft2 if desired, but 
    # generally it seems slower.
    # Opening a file for writing is supposed to truncate it
    fh = open( rampath + "/pyFFTW_wisdom.pkl", 'wb')
    pickle.dump( pyfftw.export_wisdom(), fh )
    fh.close()
    return FFT2, IFFT2
    
@static_var( "bpFilter", -1 )
@static_var( "mageShape", np.array([0,0]) )
@static_var( "ps", -42 )
@static_var( "FFT2", -42 )
@static_var( "IFFT2", -42 )
def IceFilter( mage, pixelSize=1.0, filtRad = 8.0 ):
    """
    IceFilter applies a band-pass filter to mage that passes the first 3 
    water ice rings, and then returns the result.
        pixelSize is in ANGSTROMS because this is bio.  Program uses this to 
        calculate the width of the band-pass filter.
        filtRad is radius of the Gaussian filter (pixels) to apply after Fourier filtration 
        that are periodic artifacts due to multiple defocus zeros being in the band 
    """
    
    # First water ring is at 3.897 Angstroms
    # Second is ater 3.669 Angstroms
    # Third is at 3.441 Angstroms
    # And of course there is strain, so go from about 4 to 3.3 Angstroms in the mesh

    # Check to see if we have to update our static variables
    if ( (IceFilter.mageShape != mage.shape).any() ) or (IceFilter.bpFilter.size == 1) or (IceFilter.ps != pixelSize):
        # Make a new IceFilter.bpFilter
        IceFilter.mageShape = np.array( mage.shape )
        IceFilter.ps = pixelSize
        
        bpMin = pixelSize / 4.0  # pixels tp the 4.0 Angstrom spacing
        bpMax = pixelSize / 3.3  # pixels to the 3.3 Angstrom spacing
        
        # So pixel frequency is -0.5 to +0.5 with shape steps
        # And we want a bandpass from 1.0/bpMin to 1.0/bpMax, which is different on each axis for rectangular images
        pixFreqX = 1.0 / mage.shape[1]
        pixFreqY = 1.0 / mage.shape[0]
        bpRangeX = np.round( np.array( [ bpMin/pixFreqX, bpMax/pixFreqX ] ) )
        bpRangeY = np.round( np.array( [ bpMin/pixFreqY, bpMax/pixFreqY ] ) )
        IceFilter.bpFilter = np.fft.fftshift( 
            (1.0 - apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[0],bpRangeX[0] ] )) 
            * apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[1],bpRangeX[1] ] ) )
        IceFilter.bpFilter = IceFilter.bpFilter.astype( 'float32' ) 
        [IceFilter.FFT2, IceFilter.IFFT2] = pyFFTWPlanner( mage.astype('complex64') )
        pass
    
    # Apply band-pass filter
    IceFilter.FFT2.update_arrays( mage.astype('complex64'), IceFilter.FFT2.get_output_array() )
    IceFilter.FFT2.execute()
    IceFilter.IFFT2.update_arrays( IceFilter.FFT2.get_output_array() * IceFilter.bpFilter, IceFilter.IFFT2.get_output_array() )
    IceFilter.IFFT2.execute()
    
    bpMage = IceFilter.IFFT2.get_output_array()
    bpMage /= bpMage.size
    bpGaussMage = scipy.ndimage.gaussian_filter( np.abs(bpMage), filtRad )
    # So if I don't want to build a mask here, and if I'm just doing band-pass
    # intensity scoring I don't need it, I don't need to make a thresholded mask
    
    # Should we normalize the bpGaussMage by the mean and std of the mage?
    return bpGaussMage
    
#### DEBUG / TESTING MAIN CODE #####

#fitbackground  = False
#testfile = "E:/CDI/2014Aug04_ProbeSeries/DiffSi200kV_10umSS7_770mm_50ms_49.dm3"
#dm3struct = dm3.DM3( testfile )
#
#mage = dm3struct.imagedata
#
#t0 = time.time()
#pk_loc = findPeaks2( mage, peak_rad=15.0, threshold = 30, showPlot=True )
#t1 = time.time()
#print "Time for findPeaks2 (s) : " + str(t1-t0)

#### DEBUG / TESTING MAIN CODE #####
