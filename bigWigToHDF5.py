#!/usr/bin/env python
"""
Script for preprocessing data with a user-supplied DNase BED file.

Use `makeData2.py -h` to see an auto-generated description of advanced options.
"""

import numpy as np
import pyBigWig
import h5py

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
        chrom_values = np.array(bw.values(chrom, 0, chrom_size), dtype=np.float32)
        if mode == 'footprinting':
            # If a value is undefined, impute with 0
            chrom_nan = np.isnan(chrom_values)
            impute_value = 0
            chrom_values[chrom_nan] = impute_value
            chrom_sum = chrom_values.sum()
            chrom_values = chrom_values / chrom_sum * chrom_size / 1000.0
        f.create_dataset(chrom, data=chrom_values, compression='gzip')
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
    parser.add_argument('--mode', '-m', choices=['none', 'footprinting'], default='none',
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
