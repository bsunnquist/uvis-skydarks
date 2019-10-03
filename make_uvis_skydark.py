#! /usr/bin/env python

"""
Calibrates input UVIS files based on user inputs. Options include subtracting
off the median of each amp from the input files (to remove any bias offset),
and/or creating and subtracting a skydark from the input files.

Authors
-------
    Ben Sunnquist, 2019

Use
---
    This code should be run in the same directory as the collection
    of UVIS flts of a given filter. 

    To run from the command line:
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
import scipy

# -----------------------------------------------------------------------------

def make_segmap(f, overwrite=True):
    """
    Makes a segmentation map for each extension of the input files.
    
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
        if (os.path.exists(outfile)) & (overwrite is False):
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
    if (os.path.exists(outfile)) & (overwrite is False):
        print('{} already exists, stopping...'.format(outfile))

    else:
        print('Making a stack of the input files...')
        stack = np.zeros((len(files), 2051, 4096))
        for i,f in enumerate(files):
            h = fits.open(f)
            data = h[ext].data
            #dq = h[ext+2].data

            # Get the segmap for this file
            segmap_file = f.replace('.fits', '_seg_ext_{}.fits'.format(ext))
            if not os.path.isfile(segmap_file):  # sometimes input files are medsub
                segmap_file = f.replace('_medsub.fits', '_seg_ext_{}.fits'.format(ext))
            segmap = fits.getdata(segmap_file)

            # Mask bad pixels and sources
            #data[dq!=0] = np.nan
            data[segmap>0] = np.nan
            stack[i] = data
            h.close()

        # Make the skydark
        print('Calculating the median through the stack of input files...')
        if nproc==1:
            skydark = np.nanmedian(stack, axis=0)
        else:
            stacks = np.split(stack, 16, axis=2)  # split stack into 16 2048x256 sections
            p = Pool(nproc)
            results = p.map(med_stack, stacks)
            skydark = np.concatenate(results, axis=1)

        # Write out the sky dark
        fits.writeto(outfile, skydark, overwrite=True)
        print('Sky dark generated.')

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
        print('Filtered sky dark generated.')

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

def subtract_med(f):
    """
    Subtracts the median from the input file amp-by-amp to remove any bias
    offset. Assumes the segmaps are in the same directory.
    
    Parameters
    ----------
    f : str
        The filename to process.

    Outputs
    -------
    {f}_medsub.fits : fits image
        The file with the median subtracted amp-by-amp
    """
    
    outfile = f.replace('.fits', '_medsub.fits')
    if not os.path.isfile(outfile):
        h = fits.open(f)
        for i in [1,4]:
            data_orig = h[i].data
            data1_orig, data2_orig = np.split(data_orig, 2, axis=1)  # split amps
            
            # Make copies of the original data with sources flagged
            data = np.copy(data_orig)
            segmap = fits.getdata(f.replace('.fits', '_seg_ext_{}.fits'.format(i)))
            data[segmap > 0] = np.nan  # flag sources
            data1, data2 = np.split(data, 2, axis=1)  # split amps
            
            # Subtract the median of each amp from the original data
            data1_new = data1_orig - np.nanmedian(data1)
            data2_new = data2_orig - np.nanmedian(data2)
            data_new = np.concatenate([data1_new, data2_new], axis=1)  # recombine amps
            h[i].data = data_new
        
        h.writeto(outfile, overwrite=True)
        h.close()
    else:
        print('{} already exists.'.format(outfile))

# -----------------------------------------------------------------------------

def subtract_skydark(f):
    """
    Subtracts the skydark from the input file.
    
    Parameters
    ----------
    f : str
        The filename to process.

    Outputs
    -------
    {f}_skydarksub.fits : fits image
        The file with the skydark subtracted
    """

    # Store these final products in a separate directory
    final_outdir = './final/'
    outfile = os.path.join(final_outdir, f.replace('.fits', '_skydarksub.fits'))
    if not os.path.isfile(outfile):
        # Get the skydark data
        skydark_ext1 = fits.getdata('skydark_ext_1_filtered.fits')
        skydark_ext4 = fits.getdata('skydark_ext_4_filtered.fits')

        # Subtract the skydark from each extension
        h = fits.open(f)
        data = np.copy(h[1].data)
        data = data - skydark_ext1
        h[1].data = data.astype('float32')
        data = np.copy(h[4].data)
        data = data - skydark_ext4
        h[4].data = data.astype('float32')
        h.writeto(outfile)
        h.close()
    else:
        print('{} already exists.'.format(outfile))

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
    outdir_help = ('The directory containing the files to stack and '
                   'where to write the output skydark to.')
    nproc_help = 'The number of processes to use for multiprocessing.'
    overwrite_skydarks_help = 'Option to overwrite existing skydarks.'
    overwrite_segmaps_help = 'Option to overwrite existing segmaps.'
    no_medsub_help = ('Option to not subtract the median from each amp before '
                      'making the skydark.')
    skydark_help = ('Option to make the skydark and subtract it from the '
                    'input files.')

    # Add the potential arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', dest='outdir', action='store', type=str, 
                        required=False, help=outdir_help)
    parser.add_argument('--np', dest='nproc', action='store', type=int, 
                        required=False, help=nproc_help)
    parser.add_argument('--overwrite_skydarks', dest='overwrite_skydarks', 
                        action='store_true', required=False, 
                        help=overwrite_skydarks_help)
    parser.add_argument('--overwrite_segmaps', dest='overwrite_segmaps', 
                        action='store_true', required=False, 
                        help=overwrite_segmaps_help)
    parser.add_argument('--no_medsub', dest='subtract_med', 
                        action='store_false', required=False, 
                        help=no_medsub_help)
    parser.add_argument('--skydark', dest='make_skydark', action='store_true', 
                        required=False, help=skydark_help)
    
    # Set defaults
    parser.set_defaults(outdir=os.getcwd(), nproc=8, overwrite_skydarks=False,
                        overwrite_segmaps=False, subtract_med=True, 
                        make_skydark=False)

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

    # Input all of the files in the current directory (should all belong 
    # to the same filter)
    files = glob.glob('*flt.fits')
    print('Found {} input files.'.format(len(files)))

    # Make segmentation maps for each input file
    print('Making segmentation maps for the input files...')
    if not args.overwrite_segmaps:
        input_files = []
        for f in files:
            if not os.path.exists(f.replace('.fits', '_seg_ext_1.fits')):
                input_files.append(f)
    else:
        input_files = files

    if len(input_files) > 0:
        p = Pool(args.nproc)
        p.map(make_segmap, input_files)

    # Subtract the median amp-by-amp from each input file
    if args.subtract_med:
        print('Subtracting the median of each amp from the input files...')
        p = Pool(args.nproc)
        p.map(subtract_med, files)
        files = [f.replace('.fits', '_medsub.fits') for f in files]

    # Make the sky dark for each UVIS SCI extension and subtract it
    # from the input files.
    if args.make_skydark:
        print('Making the sky dark for extension 1...')
        make_skydark(files, ext=1, nproc=args.nproc, title='ext_1', 
                     overwrite=args.overwrite_skydarks)
        print('Sky dark complete for extension 1')
        print('Making the sky dark for extension 4...')
        make_skydark(files, ext=4, nproc=args.nproc, title='ext_4', 
                     overwrite=args.overwrite_skydarks)
        print('Sky dark complete for extension 4')

        # Make directory to store final products
        final_outdir = './final/'
        if not os.path.exists(final_outdir):
            os.mkdir(final_outdir)
        
        # Subtract the skydark from each input file
        print('Subtracting the skydark from each input file...')
        p = Pool(args.nproc)
        p.map(subtract_skydark, files)

        # Rename all final files to flcs
        print('Renaming all final products to flcs...')
        for f in glob.glob('./final/*.fits'):
            if args.subtract_med:
                os.rename(f, f.replace('_flt_medsub_skydarksub.fits', 
                                       '_flc.fits'))
            else:
                os.rename(f, f.replace('_flt_skydarksub.fits', '_flc.fits'))

    print('make_uvis_skydark.py complete.')
