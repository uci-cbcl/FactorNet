#!/usr/bin/env python
"""
Script for preprocessing data with a user-supplied bigWig file. Also imputes missing values and
normalizes values.

Use `bigWigToHDF5.py -h` to see an auto-generated description of advanced options.
"""

import numpy as np
import pyBigWig
import h5py
from scipy.ndimage.filters import uniform_filter1d

# Standard library imports
import sys
import os
import errno
import argparse


def convert(bigwig_filename, output_filename, mode):
    print 'Reading bigWig'
    bw = pyBigWig.open(bigwig_filename)
    chroms = bw.chroms()
    f = h5py.File(output_filename, 'w')
    for chrom in chroms:
        print 'Processing chromosome:', chrom
        chrom_size = chroms[chrom]
        chrom_values = np.array(bw.values(chrom, 0, chrom_size))
        if mode == 'chrommean':
            # If a value is undefined, impute with 0
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
            chrom_sum = chrom_values.sum()
            chrom_values = chrom_values / chrom_sum * chrom_size / 1000.0
        if mode == 'localzscore':  # z-score localized normalization
            # If a value is undefined, impute with 0
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
            filter_length = 1000001
            chrom_local_means = uniform_filter1d(chrom_values, filter_length, mode='constant')
            chrom_local_sqmeans = uniform_filter1d(chrom_values**2, filter_length, mode='constant')
            chrom_local_stds = np.sqrt(chrom_local_sqmeans - chrom_local_means**2)
            chrom_values = (chrom_values - chrom_local_means) / chrom_local_stds
            # Replace any nans with zeroes
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
        if mode == 'localmean':  # localized normalization, divide by mean
            # If a value is undefined, impute with 0
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
            filter_length = 1000001
            chrom_local_means = uniform_filter1d(chrom_values, filter_length, mode='constant')
            chrom_values = chrom_values / chrom_local_means / 1000.0
            # Replace any nans with zeroes
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
        f.create_dataset(chrom, data=chrom_values.astype(np.float32), compression='gzip')
    f.close()
    bw.close()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Converts and normalizes a bigWig file to an HDF5 file.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--bigwig', '-w', type=str, required=True,
                        help='1D bigWig files.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='The output file.')
    parser.add_argument('--mode', '-m', choices=['none', 'chrommean', 'localzscore', 'localmean'],
                        default='none',
                        help='How to normalize the data.')
    return parser


def main():
    """
    The main executable function
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    bigwig_filename = args.bigwig
    mode = args.mode
    output_filename = args.output

    convert(bigwig_filename, output_filename, mode)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
