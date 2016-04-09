#!/usr/bin/env python
"""
Script for preprocessing data.

Use `makeData.py -h` to see an auto-generated description of advanced options.
"""

import numpy as np
from pybedtools import BedTool
import pyfasta
import pyBigWig
import h5py

# Standard library imports
import sys
import os
import errno
import argparse


def save_data(positive_bed, x, y, output_dir, clobber):
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


def get_data(genome, fasta, beds, excludes, bigwigs, recurrent):
    num_beds = len(beds)
    num_bigwigs = len(bigwigs)
    num_channels = 4 + num_bigwigs
    num_targets = num_beds
    seq_length = 1000
    window = BedTool()
    genome_windows = window.window_maker(g=genome, w=200, s=200)
    # Exclude all windows that overlap an exclude region
    genome_windows = genome_windows.intersect(excludes, wa=True, v=True)
    # Get a bed file that overlaps at least one interval
    positive_bed = genome_windows.intersect(beds,e=True,f=0.5,F=0.5,wa=True,c=True)
    positive_bed = positive_bed.filter(lambda x: x.count > 0).saveas()
    num_samples = len(positive_bed)
    x = np.zeros((num_samples, seq_length, num_channels), dtype=np.float32)
    y = np.zeros((num_samples, num_targets), dtype=bool)
    # Generate targets
    for t in xrange(num_targets):
        bed_count = positive_bed.intersect(beds[t], e=True, f=0.5, F=0.5, wa=True, c=True)
        y[:,t] = [i.count for i in bed_count]
    # Generate features
    positive_slop_bed = positive_bed.slop(g=genome, b=400)
    fasta = pyfasta.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwigs]
    d = np.array(['A', 'C', 'G', 'T'])
    for n in xrange(num_samples):
        interval = positive_slop_bed[n]
        # Get FASTA string sequence first
        s = np.array(list(fasta[interval.chrom][interval.start:interval.stop].upper()))
        x[n, :, 0:4] = s[:,np.newaxis] == d
        for f in xrange(num_bigwigs):
            bw = bigwigs[f]
            x[n, :, f+4] = bw.values(interval.chrom, interval.start, interval.stop)

    for bigwig in bigwigs:
        bigwig.close()
    return positive_bed, x, y


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Generate training, validation, and testing sets.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--fasta', '-f', type=str, required=True,
                        help='Genome fasta file')
    parser.add_argument('--genome', '-g', type=str, required=True,
                        help='Genome size file.')
    parser.add_argument('--bed', '-b', type=str, required=True, nargs='+',
                        help='One or more BED files.')
    parser.add_argument('--exclude', '-x', type=str, required=False, nargs='+',
                        help='One or more BED files of regions to exclude.')
    parser.add_argument('--bigwig', '-w', type=str, required=False, nargs='*',
                        help='1D bigWig files (optional).')
    parser.add_argument('--recurrent', '-r', action='store_true',
                        help='Save data as long sequences for fully recurrent models.')
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

    genome = args.genome
    fasta = args.fasta
    beds = args.bed
    bigwigs = args.bigwig
    excludes = args.exclude
    recurrent = args.recurrent

    positive_bed, x, y = get_data(genome, fasta, beds, excludes, bigwigs, recurrent)

    if args.outputdir is None:
        clobber = True
        output_dir = args.outputdirc
    else:
        clobber = False
        output_dir = args.outputdir
    save_data(positive_bed, x, y, output_dir, clobber)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
