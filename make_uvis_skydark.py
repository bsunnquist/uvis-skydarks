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
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
import glob
from multiprocessing import Pool
import numpy as np
import os
from photutils import detect_sources, detect_threshold, Background2D, MedianBackground
from scipy.signal import medfilt

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
            segm = detect_sources(data, threshold, npixels=3, filter_kernel=kernel)
            fits.writeto(outfile, segm.data, overwrite=overwrite)

# -----------------------------------------------------------------------------

def make_skydark(files, ext=1, nproc=6, title='ext_1', overwrite=False):
    """
    Makes a UVIS sky dark.
    
    Parameters
    ----------
    files : list
        The files to use to make the sky dark.

    ext : int
        The UVIS SCI extension to make the sky dark for.

    np : int
        The number of processes to use when stacking.

    title : str
        The title to pad to the output sky dark filename.

    overwrite : bool
        Option to overwrite existing skydarks.

    Outputs
    -------
    skydark_{title}.fits
        The sky dark.
    
    skydark_{title}_filtered.fits
        A filtered version of the sky dark.
    """

    # See if outfile already exists
    outfile = 'skydark_{}.fits'.format(title)
    if (os.path.exists(outfile)) & (overwrite==False):
        print('{} already exists, stopping...'.format(outfile))

    else:
        # Make a stack of all input data
        print('Making a stack of the input flcs...')
        stack = np.zeros((len(files), 2051, 4096))
        for i,f in enumerate(files):
            h = fits.open(f)
            data = h[ext].data
            #dq = h[ext+2].data
            segmap = fits.getdata(f.replace('.fits', '_seg_ext_{}.fits'.format(ext)))

            # mask bad pixels and sources
            #data[dq!=0] = np.nan
            data[segmap>0] = np.nan
            stack[i] = data
            h.close()

        # Make the skydark
        print('Calculating the median through the stack of flcs...')
        if nproc==1:
            skydark = np.nanmedian(stack, axis=0)
        else:
            stacks = np.split(stack, 16, axis=2)  # split stack into 16 2048x256 sections
            p = Pool(nproc)
            results = p.map(med_stack, stacks)
            skydark = np.concatenate(results, axis=1)

        # Write out the sky dark
        fits.writeto(outfile, skydark, overwrite=True)
        print('Sky dark generated')

        # Make a filtered version of the skydark
        print('Filtering the sky dark...')
        amp1, amp2 = np.split(skydark, 2, axis=1)  # treat amps separately
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = MedianBackground()
        bkg1 = Background2D(amp1, (100, 100), filter_size=(10, 10), 
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bkg2 = Background2D(amp2, (100, 100), filter_size=(10, 10), 
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        filtered = np.concatenate((bkg1.background, bkg2.background), axis=1)
        fits.writeto('{}_filtered.fits'.format(outfile.replace('.fits','')), 
                     filtered, overwrite=True)
        print('Filtered sky dark generated')

# -----------------------------------------------------------------------------

def med_stack(stack):
    """
    Find the median through an image stack. Useful for multiprocessing when 
    running into memory issues.
    
    Parameters
    ----------
    stack : np.array
        A 3D stack of 2D image arrays.
    
    Returns
    -------
    med : float
        The median value through the stack.
    """

    return np.nanmedian(stack, axis=0)

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

    # Make the help strings
    outdir_help = ('The directory containing the flcs to stack and '
                   'where to write the output skydark to.')
    nproc_help = 'The number of processes to use for multiprocessing.'
    overwrite_skydarks_help = 'Option to overwrite existing skydarks.'
    overwrite_segmaps_help = 'Option to overwrite existing segmaps.'

    # Add the potential arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', dest='outdir', action='store', type=str, 
                        required=False, help=outdir_help)
    parser.add_argument('--np', dest='nproc', action='store', type=int, 
                        required=False, help=nproc_help)
    parser.add_argument('--overwrite_skydarks', dest='overwrite_skydarks', 
                        action='store_true',  required=False, 
                        help=overwrite_skydarks_help)
    parser.add_argument('--overwrite_segmaps', dest='overwrite_segmaps', 
                        action='store_true',  required=False, 
                        help=overwrite_segmaps_help)
    
    # Set defaults
    parser.set_defaults(outdir=os.getcwd(), nproc=6, overwrite_skydarks=False,
                        overwrite_segmaps=False)

    # Get the arguments
    args = parser.parse_args()

    return args

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Get the command line arguments
    args = parse_args()

    # Go to the working directory
    os.chdir(args.outdir)

    # Input all of the flc files in the current directory (should all belong 
    # to the same filter)
    files = glob.glob('*flc.fits')

    # Make segmentation maps for each input flc
    print('Making segmentation maps for the input files...')
    if args.overwrite_segmaps==False:
        input_files = []
        for f in files:
            if not os.path.exists(f.replace('.fits', '_seg_ext_1.fits')):
                input_files.append(f)
    else:
        input_files = files

    if len(input_files) > 0:
        p = Pool(args.nproc)
        p.map(make_segmap, input_files)

    # Make the sky dark for each UVIS SCI extension
    print('Making the sky dark for extension 1...')
    make_skydark(files, ext=1, nproc=args.nproc, title='ext_1', 
                 overwrite=args.overwrite_skydarks)
    print('Sky dark complete for extension 1')
    print('Making the sky dark for extension 4...')
    make_skydark(files, ext=4, nproc=args.nproc, title='ext_4', 
                 overwrite=args.overwrite_skydarks)
    print('Sky dark complete for extension 4')
