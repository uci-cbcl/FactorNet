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


def save_data(positive_bed, x_seq, x_bws, y, output_dir, clobber):
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
    print 'Saving bed'
    positive_bed.saveas(output_dir + '/windows.bed.gz')
    print 'Saving data'
    f = h5py.File(output_dir + '/data.hdf5', 'w')
    f.create_dataset('x_seq', data=x_seq, compression='gzip')
    f.create_dataset('x_bws', data=x_bws, compression='gzip')
    f.create_dataset('y', data=y, compression='gzip')
    f.close()


def get_chrom_features(data):
    interval = data[0]
    chrom_s = data[1]
    chrom_bws = data[2]
    num_bigwigs = len(chrom_bws)
    x = np.zeros((1000, 4 + num_bigwigs))
    x[:, 0:4] = chrom_s[interval.start:interval.stop, :]
    for f in xrange(num_bigwigs):
        x[:, f + 4] = chrom_bws[f][interval.start:interval.stop]
    return x


def get_overlaps(beds):
    peak_bed = beds[0]
    positive_bed = BedTool(beds[1])
    bed_count = positive_bed.intersect(peak_bed, e=True, f=0.5,
                                       F=0.5, wa=True, c=True, sorted=True)
    overlaps = [i.count > 0 for i in bed_count]
    return overlaps


def get_data(genome, fasta, beds, excludes, bigwigs, recurrent):
    assert beds  # Make sure at least one bed file is present
    chroms = np.loadtxt(genome, usecols=[0], dtype=str)
    chroms_lens = np.loadtxt(genome, usecols=[1], dtype=np.int64)
    chroms_lens_dict = dict(zip(chroms, chroms_lens))
    num_beds = len(beds)
    num_bigwigs = len(bigwigs)
    num_channels = 4 + num_bigwigs
    num_targets = num_beds
    seq_length = 1000
    window = BedTool()
    print 'Windowing genome'
    genome_windows = window.window_maker(g=genome, w=200, s=200)
    # Exclude all windows that overlap an exclude region
    if excludes:
        print 'Removing excluded regions'
        genome_windows = genome_windows.intersect(excludes, wa=True, v=True)
    # Get a bed file that overlaps at least one interval
    print 'Sorting BED files'
    beds = [BedTool(bed).sort() for bed in beds]
    print 'Intersecting genome windows with bed files'
    merged_targets_bed = beds[0]
    if len(beds) > 1:
        merged_targets_bed = merged_targets_bed.cat(*beds[1:])
    #merged_targets_bed = merged_targets_bed.slop(g=genome, b=1000)
    positive_bed = genome_windows.intersect(merged_targets_bed, u=True).sort()
    num_samples = len(positive_bed)
    x_seq = np.zeros((num_samples, seq_length, 4), dtype=bool)
    x_bws = np.zeros((num_samples, seq_length, num_bigwigs), dtype=np.float32)
    y = np.zeros((num_samples, num_targets), dtype=bool)
    print 'Number of samples:', num_samples
    print 'Number of targets:', num_targets
    print 'Number of channels:', num_channels
    # Generate targets
    print 'Generating target array'
    for t in xrange(num_targets):
        bed_count = positive_bed.intersect(beds[t], e=True, f=0.5, F=0.5, wa=True, c=True,
                                           sorted=True)
        y[:, t] = [i.count for i in bed_count]
    # Generate features
    positive_slop_bed = positive_bed.slop(g=genome, b=400)
    fasta = pyfasta.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwigs]
    d = np.array(['A', 'C', 'G', 'T'])
    print 'Generating features array'
    chrom = None
    for n, interval in enumerate(positive_slop_bed):
        if chrom != interval.chrom:
            chrom = interval.chrom
            print chrom
            chrom_size = chroms_lens_dict[chrom]
            # Grab intervals on this chromosome
            # chrom_intervals = positive_slop_bed.filter(lambda x: x.chrom == chrom)
            # chrom_intervals = list(positive_slop_bed.filter(lambda x: x.chrom == chrom).saveas())
            # num_chrom_intervals = len(chrom_intervals)
            # Grab FASTA string sequence and convert to one hot matrix
            print '\tConverting string sequence to one-hot matrix'
            chrom_s = np.array(list(fasta[chrom][0:chrom_size].upper()))[:,np.newaxis] == d
            # Grab chromosome sequence arrays for each big wig
            chrom_bws = np.zeros((chrom_size, num_bigwigs))
            print '\tConverting bigwig chromosome sequences to arrays'
            for f, bw in enumerate(bigwigs):
                chrom_bw = np.array(bw.values(chrom, 0, chrom_size))
                # If a value is undefined, impute with
                # chromosome average (minority undefined) or 0 (majority undefined)
                chrom_bw_nan = np.isnan(chrom_bw)
                if chrom_bw_nan.sum() > len(chrom_bw_nan)/2.0:
                    impute_value = 0
                else:
                    impute_value = np.mean(chrom_bw[~chrom_bw_nan])
                chrom_bw[chrom_bw_nan] = impute_value
                chrom_bws[:, f] = chrom_bw
            print '\tAdding chromosome data sequences to feature tensor'
        x_seq[n, :, 0:4] = chrom_s[interval.start:interval.stop, :]
        x_bws[n, :, :] = chrom_bws[interval.start:interval.stop]

    for bigwig in bigwigs:
        bigwig.close()
    return positive_bed, x_seq, x_bws, y


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

    positive_bed, x_seq, x_bws, y = get_data(genome, fasta, beds, excludes, bigwigs, recurrent)

    if args.outputdir is None:
        clobber = True
        output_dir = args.outputdirc
    else:
        clobber = False
        output_dir = args.outputdir
    save_data(positive_bed, x_seq, x_bws, y, output_dir, clobber)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
