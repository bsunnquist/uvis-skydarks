#! /usr/bin/env python

"""
Creates UVIS sky darks for a given filter.

Authors
-------
    Ben Sunnquist, 2019

Use
---
    This code should be run in the same directory as the collection
    of UVIS flcs of a given filter. 

    To make the sky dark:
    >>> python make_uvis_skydark.py

Notes
-----
    Written in Python 3.
"""

import argparse
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
import glob
import numpy as np
import os
from photutils import detect_sources, detect_threshold

# -----------------------------------------------------------------------------

def make_segmap(f, overwrite=True):
    """
    Makes a segmentation map for each extension of the input UVIS flc
    
    Parameters
    ----------
    f : str
        The filename to make segmentation maps for.

    overwrite : bool
        Option to overwrite existing segmaps if they exist.
    
    Outputs
    -------
    {f}_seg_ext_1.fits
        The segmentation map for SCI extension 1.

    {f}_seg_ext_4.fits
        The segmentation map for SCI extension 4.
    """
    
    # Make segmaps for each SCI extension
    for i in [1,4]:
        # See if segmap already exists
        outfile = f.replace('.fits', '_seg_ext_{}.fits'.format(i))
        if (os.path.exists(outfile)) & (overwrite==False):
            pass

        else:
            # Get the data
            data = fits.getdata(f,i)
           
            # Detector sources; Make segmap
            threshold = detect_threshold(data, snr=1.0)
            sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
            kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
            kernel.normalize()
            segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
            fits.writeto(outfile, segm.data, overwrite=overwrite)

# -----------------------------------------------------------------------------

def make_skydark(files, ext=1, title='ext_1'):
    """
    Makes a UVIS sky dark.
    
    Parameters
    ----------
    files : list
        The files to use to make the sky dark.

    ext : int
        The UVIS SCI extension to make the sky dark for

    title : str
        The title to pad to the output sky dark filename
    
    Outputs
    -------
    skydark_{title}.fits
        The sky dark
    """

    # Make a stack of all input data
    stack = np.zeros((len(files), 2051, 4096))
    for f in files:
        data = fits.getdata(f, ext)
        segmap = fits.getdata(f.replace('.fits', '_seg_ext_{}.fits'.format(ext)))
        data[segmap>0] = np.nan

    # Make the skydark
    skydark = np.nanmedian(stack, axis=0)

    # Write out the sky dark
    fits.writeto('skydark_{}.fits'.format(title))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    """
    Parses command line arguments.
    
    Returns
    -------
    args : object
        Contains the input arguments.
    """

    parser = argparse.ArgumentParser()

    # Get the arguments
    args = parser.parse_args()

    return args

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Get the command line arguments
    args = parse_args()

    # Go to the working directory
    os.chdir(args.directory)

    # Input all of the flc files in the current directory (should all belong 
    # to the same filter)
    files = glob.glob('*.fits')

    # Make segmentation maps for each input flc
    print('Making segmentation maps for the input files...')
    for f in files:
        if not os.path.exists(f.replace('.fits', '_seg_ext_1.fits')):
            make_segmap(f)

    # Make the sky dark for each UVIS SCI extension
    print('Making the sky darks...')
    make_skydark(files, ext=1, title='ext_1')
    print('sky dark Complete for extension 1')
    make_skydark(files, ext=4, title='ext_4')
    print('sky dark Complete for extension 4')
