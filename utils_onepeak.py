import numpy as np
from pybedtools import BedTool, Interval
import pyfasta
import parmap
import utils

# Standard library imports
import os
import itertools
import random

L = utils.L
w = utils.w
w2 = utils.w2
batch_size = utils.batch_size
seed = utils.seed
genome_sizes_file = utils.genome_sizes_file

genome_window_size = 200


def data_to_bed(data):
    intervals = []
    for datum in data:
        chrom = datum[0]
        start = datum[1]
        stop = datum[2]
        intervals.append(Interval(chrom, start, stop))
    return BedTool(intervals)


def extract_data_from_bed(args, shift, label):
    peaks = args[0]
    bigwig_files = args[1]

    data = []

    for peak in peaks:
        chrom = peak.chrom
        peak_start = peak.start
        peak_stop = peak.stop
        peak_mid = (peak_start + peak_stop)/2
        start = peak_mid - genome_window_size/2
        stop = peak_mid + genome_window_size/2
        if shift:
            shift_size = peak_stop - start - 75 - 1
        else:
            shift_size = 0
        data.append((chrom, start, stop, shift_size, bigwig_files, label))

    return data


def valid_test_split_wrapper(bed, valid_chroms, test_chroms):
    bed_train = []
    bed_valid = []
    bed_test = []
    for interval in bed:
        chrom = interval.chrom
        start = interval.start
        stop = interval.stop
        if chrom in test_chroms:
            bed_test.append(interval)
        elif chrom in valid_chroms:
            bed_valid.append(interval)
        else:
            bed_train.append(interval)
    bed_train = BedTool(bed_train)
    bed_valid = BedTool(bed_valid)
    bed_test = BedTool(bed_test)
    return bed_train, bed_valid, bed_test


def negative_shuffle_wrapper(args, include_bed, num_copies, noOverlapping):
    positive_windows = args[0]
    nonnegative_regions_bed = args[1]
    bigwig_files = args[2]
    if num_copies > 1:
        positive_windows = BedTool.cat(*(num_copies * [positive_windows]), postmerge=False)
    negative_windows = positive_windows.shuffle(g=genome_sizes_file,
                                                incl=include_bed.fn,
                                                excl=nonnegative_regions_bed.fn,
                                                noOverlapping=noOverlapping)
    return negative_windows


def make_features(chip_bed_list, nonnegative_regions_bed_list, bigwig_files_list, bigwig_names,
                  genome, epochs, negatives, valid_chroms, test_chroms):
    chroms, chroms_sizes, genome_bed = utils.get_genome_bed()
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    genome_bed_train, genome_bed_valid, genome_bed_test = \
        [utils.subset_chroms(chroms_set, genome_bed) for chroms_set in
         (train_chroms, valid_chroms, test_chroms)]

    print 'Splitting ChIP peaks into training, validation, and testing BEDs'
    chip_bed_split_list = parmap.map(valid_test_split_wrapper, chip_bed_list, valid_chroms, test_chroms)
    chip_bed_train_list, chip_bed_valid_list, chip_bed_test_list = zip(*chip_bed_split_list)

    positive_label = [True]
    #Train
    print 'Extracting data from positive training BEDs'
    positive_data_train_list = parmap.map(extract_data_from_bed,
                                          zip(chip_bed_train_list, bigwig_files_list),
                                          True, positive_label)
    positive_data_train = list(itertools.chain(*positive_data_train_list))

    #Validation
    print 'Extracting data from positive validation BEDs'
    positive_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(chip_bed_valid_list, bigwig_files_list),
                                          False, positive_label)
    positive_data_valid = list(itertools.chain(*positive_data_valid_list))

    #Testing
    print 'Extracting data from positive testing BEDs'
    positive_data_test_list = parmap.map(extract_data_from_bed,
                                         zip(chip_bed_test_list, bigwig_files_list),
                                         False, positive_label)
    positive_data_test = list(itertools.chain(*positive_data_test_list))

    print 'Shuffling positive training windows in negative regions'
    train_noOverlap = True
    positive_windows_train_list = parmap.map(data_to_bed, positive_data_train_list)
    negative_windows_train_list = parmap.map(negative_shuffle_wrapper,
                                             zip(positive_windows_train_list, nonnegative_regions_bed_list,
                                                 bigwig_files_list),
                                             genome_bed_train, negatives*epochs, train_noOverlap)

    print 'Shuffling positive validation windows in negative regions'
    positive_windows_valid_list = parmap.map(data_to_bed, positive_data_valid_list)
    negative_windows_valid_list = parmap.map(negative_shuffle_wrapper,
                                             zip(positive_windows_valid_list, nonnegative_regions_bed_list,
                                                 bigwig_files_list),
                                             genome_bed_valid, negatives, True)

    print 'Shuffling positive testing windows in negative regions'
    positive_windows_test_list = parmap.map(data_to_bed, positive_data_test_list)
    negative_windows_test_list = parmap.map(negative_shuffle_wrapper,
                                            zip(positive_windows_test_list, nonnegative_regions_bed_list,
                                                bigwig_files_list),
                                            genome_bed_test, negatives, True)

    negative_label = [False]
    #Train
    print 'Extracting data from negative training BEDs'
    negative_data_train_list = parmap.map(extract_data_from_bed,
                                          zip(negative_windows_train_list, bigwig_files_list),
                                          False, negative_label)
    negative_data_train = list(itertools.chain(*negative_data_train_list))

    #Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(negative_windows_valid_list, bigwig_files_list),
                                          False, negative_label)
    negative_data_valid = list(itertools.chain(*negative_data_valid_list))

    #Testing
    print 'Extracting data from negative testing BEDs'
    negative_data_test_list = parmap.map(extract_data_from_bed,
                                         zip(negative_windows_test_list, bigwig_files_list),
                                         False, negative_label)
    negative_data_test = list(itertools.chain(*negative_data_test_list))

    data_valid = negative_data_valid + positive_data_valid
    data_test = negative_data_test + positive_data_test

    print 'Shuffling training data'
    num_negatives_per_epoch = negatives*len(positive_data_train)
    np.random.shuffle(negative_data_train)
    data_train = []
    for i in xrange(epochs):
        epoch_data = []
        epoch_data.extend(positive_data_train)
        epoch_data.extend(negative_data_train[i*num_negatives_per_epoch:(i+1)*num_negatives_per_epoch])
        np.random.shuffle(epoch_data)
        data_train.extend(epoch_data)

    print 'Generating data iterators'
    from iter_onepeak import DataIterator
    bigwig_rc_order = utils.get_bigwig_rc_order(bigwig_names)
    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order)
    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order, shuffle=True)
    datagen_test = DataIterator(data_test, genome, batch_size, L, bigwig_rc_order)

    print len(datagen_train), 'training samples'
    print len(datagen_valid), 'validation samples'
    print len(datagen_test), 'testing samples'
    return datagen_train, datagen_valid, datagen_test


def load_genome():
    return utils.load_genome()

def nonnegative_wrapper(a, bl_file):
    bl = BedTool(bl_file)
    a_slop = a.slop(g=genome_sizes_file, b=genome_window_size)
    return bl.cat(a_slop).fn


def get_chip_bed(input_dir, tf, bl_file):
    blacklist = BedTool(bl_file)
    chip_info_file = input_dir + '/chip.txt'
    chip_info = np.loadtxt(chip_info_file, dtype=str)
    if len(chip_info.shape) == 1:
        chip_info = np.reshape(chip_info, (-1,len(chip_info)))
    tfs = list(chip_info[:, 1])
    assert tf in tfs
    tf_index = tfs.index(tf)
    chip_bed_file = input_dir + '/' + chip_info[tf_index, 0]
    chip_bed = BedTool(chip_bed_file)
    chip_bed = chip_bed.sort()
    #Remove any peaks not in autosomes or X chromosome
    chroms, chroms_sizes, genome_bed = utils.get_genome_bed()
    chip_bed = chip_bed.intersect(genome_bed, u=True, sorted=True)
    #Remove any peaks in blacklist regions
    chip_bed = chip_bed.intersect(blacklist, wa=True, v=True, sorted=True)
    if chip_info.shape[1] == 3:
        relaxed_bed_file = input_dir + '/' + chip_info[tf_index, 2]
        relaxed_bed = BedTool(relaxed_bed_file)
        relaxed_bed = relaxed_bed.sort()
    else:
        relaxed_bed = chip_bed
    return chip_bed, relaxed_bed


def load_chip(input_dirs, tf):
    blacklist = utils.make_blacklist()

    print 'Loading and sorting BED file(s)'
    chip_bed_list, relaxed_bed_list = zip(*parmap.map(get_chip_bed, input_dirs, tf, blacklist.fn))

    # Later we want to gather negative windows from the genome that do not overlap
    # with a blacklisted or ChIP region
    print 'Generating regions to exclude for negative windows'
    nonnegative_regions_bed_file_list = parmap.map(nonnegative_wrapper, relaxed_bed_list, blacklist.fn)
    nonnegative_regions_bed_list = [BedTool(i) for i in nonnegative_regions_bed_file_list]
    return chip_bed_list, nonnegative_regions_bed_list


def load_bigwigs(input_dirs):
    bigwig_names = None
    bigwig_files_list = []
    for input_dir in input_dirs:
        input_bigwig_info_file = input_dir + '/bigwig.txt'
        if not os.path.isfile(input_bigwig_info_file):
            input_bigwig_names = []
            input_bigwig_files_list = []
        else:
            input_bigwig_info = np.loadtxt(input_bigwig_info_file, dtype=str)
            if len(input_bigwig_info.shape) == 1:
                input_bigwig_info = np.reshape(input_bigwig_info, (-1,2))
            input_bigwig_names = list(input_bigwig_info[:, 1])
            input_bigwig_files = [input_dir + '/' + i for i in input_bigwig_info[:,0]]
        if bigwig_names is None:
            bigwig_names = input_bigwig_names
        else:
            assert bigwig_names == input_bigwig_names
        bigwig_files_list.append(input_bigwig_files)
    return bigwig_names, bigwig_files_list


def make_model(num_bws, num_motifs, num_recurrent, num_dense):
    return utils.make_model(1, num_bws, num_motifs, num_recurrent, num_dense)
