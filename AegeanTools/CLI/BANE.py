#! /usr/bin/env python

# standard imports
import argparse
import os

from AegeanTools import BANE, __citation__

__author__ = 'Paul Hancock'


def main(argv=()):
    parser = argparse.ArgumentParser(prog='BANE', prefix_chars='-')
    parser.add_argument('image', nargs='?', default=None)
    group1 = parser.add_argument_group('Configuration Options')
    group1.add_argument("--out", dest='out_base', type=str,
                        help="Basename for output images default: "
                             "FileName_{bkg,rms}.fits")
    group1.add_argument('--grid', dest='step_size', type=int, nargs=2,
                        help='The [x,y] size of the grid to use. '
                             'Default = ~4* beam size square.')
    group1.add_argument('--box', dest='box_size', type=int, nargs=2,
                        help='The [x,y] size of the box over which the '
                             'rms/bkg is calculated. Default = 5*grid.')
    group1.add_argument('--cores', dest='cores', type=int,
                        help='Number of cores to use. '
                             'Default = all available.')
    group1.add_argument('--stripes', dest='stripes', type=int, default=None,
                        help='Number of slices.')
    group1.add_argument('--slice', dest='cube_index', type=int, default=0,
                        help='If the input data is a cube, then this slice '
                             'will determine the array index of the image '
                             'which will be processed by BANE')
    group1.add_argument('--nomask', dest='mask', action='store_false',
                        default=True,
                        help="Don't mask the output array [default = mask]")
    group1.add_argument('--noclobber', dest='clobber', action='store_false',
                        default=True,
                        help="Don't run if output files already exist. "
                             "Default is to run+overwrite.")
    group1.add_argument('--debug', dest='debug',
                        action='store_true', help='debug mode, default=False')
    group1.add_argument('--compress', dest='compress', action='store_true',
                        default=False,
                        help='Produce a compressed output file.')
    group1.add_argument('--cite', dest='cite', action="store_true",
                        default=False,
                        help='Show citation information.')

    parser.set_defaults(out_base=None, step_size=None, box_size=None,
                        twopass=True, cores=None, usescipy=False, debug=False)

    options = parser.parse_args(args=argv)

    if options.cite:
        print(__citation__)
        return 0

    if options.image is None:
        parser.print_help()
        return 0

    # Get the BANE logger.
    logging = BANE.logging
    logging_level = logging.DEBUG if options.debug else logging.INFO
    logging.basicConfig(level=logging_level,
                        format="%(process)d:%(levelname)s %(message)s")
    logging.info(
        "This is BANE {0}-({1})".format(BANE.__version__, BANE.__date__))

    if not os.path.exists(options.image):
        logging.error("File not found: {0} ".format(options.image))
        return 1

    if options.out_base is None:
        options.out_base = os.path.splitext(options.image)[0]

    if not options.clobber:
        bkgout = options.out_base + '_bkg.fits'
        rmsout = options.out_base + '_rms.fits'
        if os.path.exists(bkgout) and os.path.exists(rmsout):
            logging.error("{0} and {1} exist and you said noclobber"
                          "".format(bkgout, rmsout))
            logging.error("Not running")
            return 1

    BANE.filter_image(im_name=options.image, out_base=options.out_base,
                      step_size=options.step_size,
                      box_size=options.box_size, cores=options.cores,
                      mask=options.mask, compressed=options.compress,
                      nslice=options.stripes,
                      cube_index=options.cube_index)
    return 0
