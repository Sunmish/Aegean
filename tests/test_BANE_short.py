#! /usr/bin/env python
"""
Test BANE.py
"""
from __future__ import print_function

from AegeanTools import BANE_short as BANE
import numpy as np
import os

__author__ = 'Paul Hancock'
__date__ = '23/08/2017'

import logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("Aegean")


def test_sigmaclip():
    """Test the sigmaclipping"""
    # normal usage case
    data = np.random.random(100)
    data[13] = np.nan
    if not len(BANE.sigmaclip(data, 3, 4, reps=4)) > 0:
        raise AssertionError()

    # test list where all elements get clipped
    if not len(BANE.sigmaclip([-10, 10], 1, 2, reps=2)) == 0:
        raise AssertionError()

    # test empty list
    if not len(BANE.sigmaclip([], 0, 3)) == 0:
        raise AssertionError()


def test_filter_image():
    """Test filter image"""
    # data = np.random.random((30, 30), dtype=np.float32)
    fname = 'tests/test_files/1904-66_SIN.fits'
    outbase = 'dlme'
    rms = outbase + '_rms.fits'
    bkg = outbase + '_bkg.fits'
    BANE.filter_image(fname, out_base=outbase, step_size=(10, 10), box_size=(100, 100), nslice=4)
    if not os.path.exists(rms):
        raise AssertionError()

    os.remove(rms)
    if not os.path.exists(bkg):
        raise AssertionError()

    os.remove(bkg)


def test_ND_images():
    """Test that ND images are treated correctly"""
    fbase = 'tests/test_files/small_{0}D.fits'
    outbase = 'dlme'
    rms = outbase + '_rms.fits'
    bkg = outbase + '_bkg.fits'
    # this should work just fine, but trigger different NAXIS checks
    for fname in [fbase.format(n) for n in [3,4]]:
        BANE.filter_image(fname, out_base=outbase, nslice=4)
        os.remove(rms)
        os.remove(bkg)

    fname = fbase.format(5)
    try:
        BANE.filter_image(fname,out_base=outbase)
    except Exception as e:
        pass
    else:
        raise AssertionError()
    

if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()
