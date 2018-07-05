#! /usr/bin/env python
"""
A striped down version of BANE that can be used for testing as part of our
Pawsey Uptake Project
"""
from __future__ import print_function

from astropy.io import fits
import os
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import sys

__author__ = ['Paul Hancock', 'Mohsin Shaikh']
__date__ = '2018-07-04'
__version__ = '0.1'

logging.basicConfig(level=logging.DEBUG)

ibkg = irms = None

def sigmaclip(arr, lo, hi, reps=3):
    """
    Perform sigma clipping on an array, ignoring non finite values.

    During each iteration return an array whose elements c obey:
    mean -std*lo < c < mean + std*hi

    where mean/std are the mean std of the input array.

    Parameters
    ----------
    arr : iterable
        An iterable array of numeric types.
    lo : float
        The negative clipping level.
    hi : float
        The positive clipping level.
    reps : int
        The number of iterations to perform.
        Default = 3.

    Returns
    -------
    clipped : numpy.array
        The clipped array.
        The clipped array may be empty!

    Notes
    -----
    Scipy v0.16 now contains a comparable method that will ignore nan/inf values.
    """
    clipped = np.array(arr)[np.isfinite(arr)]

    if len(clipped) < 1:
        return clipped

    std = np.std(clipped)
    mean = np.mean(clipped)
    for _ in range(int(reps)):
        clipped = clipped[np.where(clipped > mean-std*lo)]
        clipped = clipped[np.where(clipped < mean+std*hi)]
        pstd = std
        if len(clipped) < 1:
            break
        std = np.std(clipped)
        mean = np.mean(clipped)
        if 2*abs(pstd-std)/(pstd+std) < 0.2:
            break
    return clipped


def sigma_filter(filename, region, step_size, box_size, shape, dobkg=True):
    """
    Calculate the background and rms for a sub region of an image. The results are
    written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : (float, float, float, float)
        Region within the fits file that is to be processed. (ymin, ymax, xmin, xmax).

    step_size : (int, int)
        The filtering step size

    box_size : (int, int)
        The size of the box over which the filter is applied (each step).

    shape : tuple
        The shape of the fits image

    dobkg : bool
        Do a background calculation. If false then only the rms is calculated. Default = True.

    Returns
    -------
    None
    """

    # Caveat emptor: The code that follows is very difficult to read.
    # xmax is not x_max, and x,y actually should be y,x
    # TODO: fix the code below so that the above comment can be removed

    ymin, ymax = region
    logging.debug('rows {0}-{1} starting'.format(ymin, ymax))

    # cut out the region of interest plus 1/2 the box size, but clip to the image size
    rmin = max(0, ymin - box_size[0]//2)
    rmax = min(shape[0], ymax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    with fits.open(filename, memmap=True) as a:
        if NAXIS == 2:
            data = a[0].section[rmin:rmax, 0:shape[1]]
        elif NAXIS == 3:
            data = a[0].section[0, rmin:rmax, 0:shape[1]]
        elif NAXIS == 4:
            data = a[0].section[0, 0, rmin:rmax, 0:shape[1]]
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    logging.debug('data size is {0}'.format(data.shape))

    def locations(step, r_min, r_max, c_min, c_max):
        """
        Generator function to iterate over a grid of r,c coords
        operates only within the given bounds
        Returns:
        r, c
        """

        rvals = list(range(r_min, r_max, step[0]))
        if rvals[-1] != r_max:
            rvals.append(r_max)
        cvals = list(range(c_min, c_max, step[1]))
        if cvals[-1] != c_max:
            cvals.append(c_max)
        # initial data
        for c in cvals:
            for r in rvals:
                yield r, c

    def box(r, c):
        """
        calculate the boundaries of the box centered at r,c
        with size = box_size
        """
        # TODO: check that / should be //
        r_min = int(max(0, r - box_size[0] / 2))
        r_max = int(min(data.shape[0] - 1, r + box_size[0] / 2))
        c_min = int(max(0, c - box_size[1] / 2))
        c_max = int(min(data.shape[1] - 1, c + box_size[1] / 2))
        return r_min, r_max, c_min, c_max

    bkg_points = []
    bkg_values = []
    rms_points = []
    rms_values = []

    for row, col in locations(step_size, ymin, ymax, 0, shape[1]):
        x_min, x_max, y_min, y_max = box(row, col)
        new = data[x_min:x_max, y_min:y_max]
        logging.debug("x,y = {0},{1}, x_min,x_max={2},{3} new.shape={4}".format(row,col,x_min, x_max,new.shape))
        new = np.ravel(new)
        new = sigmaclip(new, 3, 3)
        # If we are left with (or started with) no data, then just move on
        if len(new) < 1:
            logging.debug("short")
            continue

        if dobkg:
            bkg = np.median(new)
            bkg_points.append((row+ymin-rmin, col))  # these coords need to be indices into the larger array
            bkg_values.append(bkg)
        rms = np.std(new)
        rms_points.append((row+ymin-rmin, col))
        rms_values.append(rms)

    # indicies of the shape we want to write to (not the shape of data)
    gx, gy = np.mgrid[ymin:ymax, 0:shape[1]]
    # If the bkg/rms calculation above didn't yield any points, then our interpolated values are all nans
    if len(rms_points) > 1:
        logging.debug("Interpolating rms")
        ifunc = LinearNDInterpolator(rms_points, rms_values)
        # force 32 bit floats
        interpolated_rms = np.array(ifunc((gx, gy)), dtype=np.float32)
        del ifunc
    else:
        logging.debug("rms is all nans")
        interpolated_rms = np.empty(gx.shape, dtype=np.float32)*np.nan

    logging.debug("Writing rms")
    offset = ymin
    for i, row in enumerate(interpolated_rms):
        irms[i+offset] = row
    # copy the image mask
    irms[ymin:ymax, :][np.where(np.isnan(data[ymin-rmin:ymax-rmin, :]))] = np.nan
    logging.debug(" .. done writing rms")

    if dobkg:
        if len(bkg_points) > 1:
            logging.debug("Interpolating bkg")
            ifunc = LinearNDInterpolator(bkg_points, bkg_values)
            interpolated_bkg = np.array(ifunc((gx, gy)), dtype=np.float32)
            del ifunc
        else:
            logging.debug("bkg is all nans")
            interpolated_bkg = np.empty(gx.shape, dtype=np.float32)*np.nan

        logging.debug("Writing bkg")
        for i, row in enumerate(interpolated_bkg):
            ibkg[i+offset] = row
        # copy the image mask
        ibkg[ymin:ymax, :][np.where(np.isnan(data[ymin-rmin:ymax-rmin, :]))] = np.nan
        logging.debug(" .. done writing bkg")
    logging.debug('rows {0}-{1} finished'.format(ymin, ymax))
    return


def filter_image(im_name, out_base, step_size=None, box_size=None, nslice=None):
    """
    Create a background and noise image from an input image.
    Resulting images are written to `outbase_bkg.fits` and `outbase_rms.fits`

    Parameters
    ----------
    im_name : str or HDUList
        Image to filter. Either a string filename or an astropy.io.fits.HDUList.
    out_base : str
        The output filename base. Will be modified to make _bkg and _rms files.
    step_size : (int,int)
        Tuple of the x,y step size in pixels
    box_size : (int,int)
        The size of the box in piexls
    nslice : int
        The image will be divided into this many horizontal stripes for processing.
        Default = None = equal to cores

    Returns
    -------
    None

    """

    global ibkg, irms

    # read the header and determine the image size
    header = fits.getheader(im_name)
    shape = (header['NAXIS2'], header['NAXIS1'])
    # setup the global memory
    ibkg = np.zeros(shape=shape)
    irms = np.zeros(shape=shape)

    if step_size is None:
        if 'BMAJ' in header and 'BMIN' in header:
            beam_size = np.sqrt(abs(header['BMAJ']*header['BMIN']))
            if 'CDELT1' in header:
                pix_scale = np.sqrt(abs(header['CDELT1']*header['CDELT2']))
            elif 'CD1_1' in header:
                pix_scale = np.sqrt(abs(header['CD1_1']*header['CD2_2']))
                if 'CD1_2' in header and 'CD2_1' in header:
                    if header['CD1_2'] != 0 or header['CD2_1']!=0:
                        logging.warning("CD1_2 and/or CD2_1 are non-zero and I don't know what to do with them")
                        logging.warning("Ingoring them")
            else:
                logging.warning("Cannot determine pixel scale, assuming 4 pixels per beam")
                pix_scale = beam_size/4.
            # default to 4x the synthesized beam width
            step_size = int(np.ceil(4*beam_size/pix_scale))
        else:
            logging.info("BMAJ and/or BMIN not in fits header.")
            logging.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
            step_size = 16
        step_size = (step_size, step_size)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6, step_size[1]*6)

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(shape))

    # Use a striped sectioning scheme
    ny = nslice

    # box widths should be multiples of the step_size, and not zero
    width_y = int(max(shape[0]/ny/step_size[1], 1) * step_size[1])

    ystart = width_y
    yend = shape[0] - shape[0] % width_y

    # locations of the box edges
    ymins = [0]
    ymins.extend(list(range(ystart, yend, width_y)))

    ymaxs = [ystart]
    ymaxs.extend(list(range(ystart+width_y, yend+1, width_y)))
    ymaxs[-1] = shape[0]

    logging.debug("ymins {0}".format(ymins))
    logging.debug("ymaxs {0}".format(ymaxs))

    for ymin, ymax in zip(ymins, ymaxs):
        region = (ymin, ymax)
        sigma_filter(filename=im_name, region=region, step_size=step_size, box_size=box_size, shape=shape)

    logging.info("done")

    # force float 32s to avoid bloated files
    ibkg = np.array(ibkg, dtype=np.float32)
    irms = np.array(irms, dtype=np.float32)

    # generate filenames
    bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])

    # add a comment to the fits header
    header['HISTORY'] = 'BANE_short {0}-({1})'.format(__version__, __date__)

    # write our bkg/rms files and copy across the header from the input file
    write_fits(ibkg, header, bkg_out)
    write_fits(irms, header, rms_out)


###
# Helper functions
###
def write_fits(data, header, file_name):
    """
    Combine data and a fits header to write a fits file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.

    header : astropy.io.fits.hduheader
        The header for the fits file.

    file_name : string
        The file to write

    Returns
    -------
    None
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, overwrite=True)
    logging.info("Wrote {0}".format(file_name))
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    filename = 'tests/test_files/1904-66_SIN.fits'
    outname = 'pup_test'
    filter_image(im_name=filename, out_base=outname, nslice=2)
