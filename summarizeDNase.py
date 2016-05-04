#!/usr/bin/env python
"""
Script for collected statistics between a DNase BED file and a collection of BEDs.

Use `summarizeDNase.py -h` to see an auto-generated description of advanced options.
"""
import numpy as np
from pybedtools import BedTool
import h5py
import pylab
import matplotlib

# Standard library imports
import sys
import os
import errno
import argparse


def summarize(i, b, output_dir):
    print 'Sorting BED files'
    beds = [BedTool(bed).sort() for bed in b]
    bed_lens = [len(bed) for bed in beds]
    dnase = BedTool(i).sort()
    print 'Intersecting with DNase peaks'
    bed_distances = []
    for bed in beds:
        closest_bed = bed.closest(dnase, d=True, t='first')
        bed_distances.append(np.array([x.count for x in closest_bed]))
    percentages = [np.sum(x == 0)*1.0/y for x, y in zip(bed_distances, bed_lens)]
    pylab.hist(percentages)
    pylab.xlabel('Percentage of peaks that overlap open chromatin peak')
    pylab.ylabel('Number of ChIP-seq BED files')
    pylab.show()
    bed_cat = np.concatenate(bed_distances)
    total_peaks = len(bed_cat)
    print 'Percent peaks overlapping DNase peaks:', np.sum(bed_cat==0)*1.0/total_peaks
    print 'Percent peaks within 50 bp of DNase Peaks:', np.sum(bed_cat<=50)*1.0/total_peaks
    print 'Percent peaks within 100 bp of DNase Peaks:', np.sum(bed_cat<=100)*1.0/total_peaks
    print 'Percent peaks within 425 bp of DNase Peaks:', np.sum(bed_cat<=425)*1.0/total_peaks
    pylab.hist(bed_cat, range=(0,10000))
    pylab.xlabel('Distance to nearest')
    pylab.ylabel('Number of ChIP-seq peaks')
    pylab.show()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Summarize open chromatin and TF peak intersections.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dnase', '-d', type=str, required=True,
                        help='Open chromatin BED file.')
    parser.add_argument('--beds', '-b', type=str, required=True, nargs='+',
                        help='ChIP-seq BED files.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--outputdir', type=str,
                       help='The output directory. Causes error if the directory already exists.')
    group.add_argument('-oc', '--outputdirc', type=str,
                       help='The output directory. Will overwrite if directory already exists.')
    return parser


def main():
    """
    The main executable function
    """
    parser = make_argument_parser()
    args = parser.parse_args()

    i = args.dnase
    b = args.beds

    if args.outputdir is None:
        clobber = True
        output_dir = args.outputdirc
    else:
        clobber = False
        output_dir = args.outputdir

    try:  # adapted from dreme.py by T. Bailey
        os.makedirs(output_dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            if not clobber:
                print >> sys.stderr, ('output directory (%s) already exists '
                                      'but you specified not to clobber it') % output_dir
                sys.exit(1)
            else:
                print >> sys.stderr, ('output directory (%s) already exists '
                                      'so it will be clobbered') % output_dir

    summarize(i, b, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
