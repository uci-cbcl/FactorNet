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
    meta = args[2]

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
        data.append((chrom, start, stop, shift_size, bigwig_files, meta, label))

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


def make_features(chip_bed_list, nonnegative_regions_bed_list, bigwig_files_list, bigwig_names, meta_list,
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
                                          zip(chip_bed_train_list, bigwig_files_list, meta_list),
                                          True, positive_label)
    positive_data_train = list(itertools.chain(*positive_data_train_list))

    #Validation
    print 'Extracting data from positive validation BEDs'
    positive_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(chip_bed_valid_list, bigwig_files_list, meta_list),
                                          False, positive_label)
    positive_data_valid = list(itertools.chain(*positive_data_valid_list))

    #Testing
    print 'Extracting data from positive testing BEDs'
    positive_data_test_list = parmap.map(extract_data_from_bed,
                                         zip(chip_bed_test_list, bigwig_files_list, meta_list),
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
                                          zip(negative_windows_train_list, bigwig_files_list, meta_list),
                                          False, negative_label)
    negative_data_train = list(itertools.chain(*negative_data_train_list))

    #Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(negative_windows_valid_list, bigwig_files_list, meta_list),
                                          False, negative_label)
    negative_data_valid = list(itertools.chain(*negative_data_valid_list))

    #Testing
    print 'Extracting data from negative testing BEDs'
    negative_data_test_list = parmap.map(extract_data_from_bed,
                                         zip(negative_windows_test_list, bigwig_files_list, meta_list),
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
    from iter_meta import DataIterator
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


def load_meta(input_dirs):
    meta_names = None
    meta_list = []
    for input_dir in input_dirs:
        input_meta_info_file = input_dir + '/meta.txt'
        if not os.path.isfile(input_meta_info_file):
            input_meta_names = []
            input_meta_list = []
        else:
            input_meta_info = np.loadtxt(input_meta_info_file, dtype=str)
            if len(input_meta_info.shape) == 1:
                input_meta_info = np.reshape(input_meta_info, (-1,2))
            input_meta_names = list(input_meta_info[:, 1])
            input_meta = input_meta_info[:,0].astype('float32')
        if meta_names is None:
            meta_names = input_meta_names
        else:
            assert meta_names == input_meta_names
        meta_list.append(input_meta)
    return meta_names, meta_list


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


def make_model(num_bws, num_meta, num_motifs, num_recurrent, num_dense):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input
    from keras.layers.convolutional import Convolution1D, MaxPooling1D
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    forward_input = Input(shape=(L, 4 + num_bws,))
    reverse_input = Input(shape=(L, 4 + num_bws,))
    meta_input = Input(shape=(num_meta,))
    hidden_layers = [
        Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                      filter_length=w, border_mode='valid', activation='relu', 
                      subsample_length=1),
        Dropout(0.1),
        TimeDistributed(Dense(num_motifs, activation='relu')),
        MaxPooling1D(pool_length=w2, stride=w2),
        Bidirectional(LSTM(num_recurrent, dropout_W=0.1, dropout_U=0.1, return_sequences=True)),
        Dropout(0.5),
        Flatten(),
        Dense(num_dense, activation='relu'),
    ]
    forward_dense = utils.get_output(forward_input, hidden_layers)     
    reverse_dense = utils.get_output(reverse_input, hidden_layers)
    forward_dense = merge([forward_dense, meta_input], mode='concat')
    reverse_dense = merge([reverse_dense, meta_input], mode='concat')
    dense2_layer = Dense(num_dense, activation='relu')
    forward_dense2 = dense2_layer(forward_dense)
    reverse_dense2 = dense2_layer(reverse_dense)
    sigmoid_layer =  Dense(1, activation='sigmoid')
    forward_output = sigmoid_layer(forward_dense2)
    reverse_output = sigmoid_layer(reverse_dense2)
    output = merge([forward_output, reverse_output], mode='ave')
    model = Model(input=[forward_input, reverse_input, meta_input], output=output)

    print 'Compiling model'
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.summary()
    
    return model

def load_testdata(tf, genome, test_chroms, input_dir):
    chip_info_file = input_dir + '/chip.txt'
    chip_info = np.loadtxt(chip_info_file, dtype=str)
    if len(chip_info.shape) == 1:
        chip_info = np.reshape(chip_info, (-1,len(chip_info)))
    tfs = list(chip_info[:, 1])
    if tf: #tf is specified, collect one ChIP file for evaluation
        assert tf in tfs
        tf_index = tfs.index(tf)
        chip_bed_file = input_dir + '/' + chip_info[tf_index,0]
        chip_bed = BedTool(chip_bed_file).sort()
        positives_only = False
        step = genome_window_size#genome_window_step
    else: #tf is not specified, collect all input chip files and merge for visualization
        _, _, chip_bed = get_chip_beds(input_dir)
        positives_only = True
        step = genome_window_size
    _, _, genome_bed = utils.get_genome_bed()
    genome_bed_test = utils.subset_chroms(test_chroms, genome_bed)
    print 'Windowing test chromosome(s)'
    genome_bed_test_windows = BedTool().window_maker(b=genome_bed_test, w=genome_window_size,
                                                     s=step)
    blacklist = utils.make_blacklist()
    print 'Removing windows overlapping blacklist regions'
    genome_bed_test_windows = genome_bed_test_windows.intersect(blacklist, wa=True, v=True,
                                                                sorted=True)
    print 'Generating labels for test chromosome(s)'
    y = utils.intersect_count(chip_bed, genome_bed_test_windows.fn)
    y = np.array(y, dtype=bool)
    print 'Generating test data iterator'
    bigwig_names, bigwig_files = utils.load_bigwigs(input_dir)
    meta_names, meta_list = load_meta([input_dir])
    meta = meta_list[0]
    if positives_only:
        data_test = [(window.chrom, window.start, window.stop, 0, bigwig_files, meta)
                     for y_sample, window in itertools.izip(y, genome_bed_test_windows)
                     if y_sample]
    else:
        data_test = [(window.chrom, window.start, window.stop, 0, bigwig_files, meta)
                     for window in genome_bed_test_windows]
    from iter_meta import DataIterator
    bigwig_rc_order = utils.get_bigwig_rc_order(bigwig_names)
    datagen_test = DataIterator(data_test, genome, 1000, L, bigwig_rc_order, shuffle=False)
    return bigwig_names, datagen_test, y


def load_beddata(tf, genome, bed_file, input_dir):
    bed = BedTool(bed_file)
    blacklist = utils.make_blacklist()
    print 'Determining which windows are valid'
    bed_intersect_blacklist_count = bed.intersect(blacklist, wa=True, c=True, sorted=True)
    nonblacklist_bools = np.array([i.count==0 for i in bed_intersect_blacklist_count])
    print 'Filtering away blacklisted windows'
    bed_filtered = bed.intersect(blacklist, wa=True, v=True, sorted=True)
    print 'Generating test data iterator'
    bigwig_names, bigwig_files = utils.load_bigwigs(input_dir)
    meta_names, meta_list = load_meta([input_dir])
    meta = meta_list[0]
    data_bed = [(window.chrom, window.start, window.stop, 0, bigwig_files, meta)
                for window in bed_filtered]
    from iter_meta import DataIterator
    bigwig_rc_order = utils.get_bigwig_rc_order(bigwig_names)
    datagen_bed = DataIterator(data_bed, genome, 1000, L, bigwig_rc_order, shuffle=False)
    return bigwig_names, datagen_bed, nonblacklist_bools

