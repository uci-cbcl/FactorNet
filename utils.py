import numpy as np
from pybedtools import BedTool, Interval
import pyfasta
import parmap

# Standard library imports
import os
import itertools

L = 1002
w = 34
w2 = w/2

batch_size = 100

genome_sizes_file = 'resources/hg19.autoX.chrom.sizes'
genome_fasta_file = 'resources/hg19.fa'
blacklist_file = 'resources/blacklist.bed.gz'

genome_window_size = 200
genome_window_step = 50
shift_size = 20


def set_seed(seed):
    np.random.seed(seed)


def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False


def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)


def get_genome_bed():
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    return chroms, chroms_sizes, genome_bed


def get_bigwig_rc_order(bigwig_names):
    assert len(set(bigwig_names)) == len(bigwig_names)
    rc_indices = np.arange(len(bigwig_names))
    for ind, bigwig_name in enumerate(bigwig_names):
        if bigwig_name[-4:] == '_fwd':
            bigwig_rc_name = bigwig_name[:-4] + '_rev'
            bigwig_rc_index = bigwig_names.index(bigwig_rc_name)
            rc_indices[bigwig_rc_index] = ind
        if bigwig_name[-4:] == '_rev':
            bigwig_rc_name = bigwig_name[:-4] + '_fwd'
            bigwig_rc_index = bigwig_names.index(bigwig_rc_name)
            rc_indices[bigwig_rc_index] = ind
    return rc_indices


def make_features(positive_windows, y_positive, nonnegative_regions_bed, bigwig_files, bigwig_names,
                  genome, epochs, valid_chroms, test_chroms):
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    genome_bed_train, genome_bed_valid, genome_bed_test = \
        [subset_chroms(chroms_set, genome_bed) for chroms_set in
         (train_chroms, valid_chroms, test_chroms)]

    positive_windows_train = []
    positive_windows_valid = []
    positive_windows_test = []
    positive_data_train = []
    positive_data_valid = []
    positive_data_test = []
    
    print 'Splitting positive windows into training, validation, and testing sets'
    for positive_window, target_array in itertools.izip(positive_windows, y_positive):
        chrom = positive_window.chrom
        start = positive_window.start
        stop = positive_window.stop
        if chrom in test_chroms:
            positive_windows_test.append(positive_window)
            positive_data_test.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))
        elif chrom in valid_chroms:
            positive_windows_valid.append(positive_window)
            positive_data_valid.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))
        else:
            positive_windows_train.append(positive_window)
            positive_data_train.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))

    positive_windows_train = BedTool(positive_windows_train)
    positive_windows_valid = BedTool(positive_windows_valid)
    positive_windows_test = BedTool(positive_windows_test)

    print 'Getting negative training examples'
    negative_windows_train = BedTool.cat(*(epochs*[positive_windows]), postmerge=False)
    negative_windows_train = negative_windows_train.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_train.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
    print 'Getting negative validation examples'
    negative_windows_valid = positive_windows_valid.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_valid.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
    print 'Getting negative testing examples'
    negative_windows_test = positive_windows_test.shuffle(g=genome_sizes_file,
                                                          incl=genome_bed_test.fn,
                                                          excl=nonnegative_regions_bed.fn,
                                                          noOverlapping=False,
                                                          seed=np.random.randint(-214783648, 2147483647))

    # Train
    print 'Extracting data from negative training BEDs'
    negative_targets = np.zeros(y_positive.shape[1])
    negative_data_train = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_train]

    # Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_valid]

    # Testing
    print 'Extracting data from negative testing BEDs'
    negative_data_test = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                          for window in negative_windows_test]
    
    num_positive_train_windows = len(positive_data_train)
    
    data_valid = negative_data_valid + positive_data_valid
    data_test = negative_data_test + positive_data_test

    print 'Shuffling training data'
    data_train = []
    for i in xrange(epochs):
        epoch_data = []
        epoch_data.extend(positive_data_train)
        epoch_data.extend(negative_data_train[i*num_positive_train_windows:(i+1)*num_positive_train_windows])
        np.random.shuffle(epoch_data)
        data_train.extend(epoch_data)

    print 'Generating data iterators'
    from iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order, shift=shift_size, shuffle=False)
    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order, shift=False, shuffle=False)
    datagen_test = DataIterator(data_test, genome, batch_size, L, bigwig_rc_order, shift=False, shuffle=False)

    print len(datagen_train), 'training samples'
    print len(datagen_valid), 'validation samples'
    print len(datagen_test), 'testing samples'
    return datagen_train, datagen_valid, datagen_test


def get_onehot_chrom(chrom): 
    fasta = pyfasta.Fasta(genome_fasta_file)
    chr_str = str(fasta[chrom]).upper()
    d = np.array(['A','C','G','T'])
    y = np.fromstring(chr_str, dtype='|S1')[:, np.newaxis] == d
    return y


def load_genome():
    chroms = list(np.loadtxt(genome_sizes_file, usecols=[0], dtype=str))
    onehot_chroms = parmap.map(get_onehot_chrom, chroms)
    genome_dict = dict(zip(chroms, onehot_chroms))
    return genome_dict


def intersect_count(chip_bed, windows_file):
    windows = BedTool(windows_file)
    chip_bedgraph = windows.intersect(chip_bed, wa=True, c=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)
    bed_counts = [i.count for i in chip_bedgraph]
    return bed_counts


def make_blacklist():
    blacklist = BedTool(blacklist_file)
    blacklist = blacklist.slop(g=genome_sizes_file, b=L)
    # Add ends of the chromosomes to the blacklist
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    blacklist2 = []
    for chrom, size in zip(chroms, chroms_sizes):
        blacklist2.append(Interval(chrom, 0, L))
        blacklist2.append(Interval(chrom, size - L, size))
    blacklist2 = BedTool(blacklist2)
    blacklist = blacklist.cat(blacklist2)
    return blacklist


def get_chip_beds(input_dir):
    chip_info_file = input_dir + '/chip.txt'
    chip_info = np.loadtxt(chip_info_file, dtype=str)
    if len(chip_info.shape) == 1:
        chip_info = np.reshape(chip_info, (-1,len(chip_info)))
    tfs = list(chip_info[:, 1])
    chip_bed_files = [input_dir + '/' + i for i in chip_info[:,0]]
    chip_beds = [BedTool(chip_bed_file) for chip_bed_file in chip_bed_files]
    print 'Sorting BED files'
    chip_beds = [chip_bed.sort() for chip_bed in chip_beds]
    if len(chip_beds) > 1:
        merged_chip_bed = BedTool.cat(*chip_beds)
    else:
        merged_chip_bed = chip_beds[0]
    return tfs, chip_beds, merged_chip_bed


def load_chip(input_dir):
    tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)
    print 'Removing peaks outside of X chromosome and autosomes'
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

    print 'Windowing genome'
    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=genome_window_size,
                                            s=genome_window_step)

    print 'Extracting windows that overlap at least one ChIP interval'
    positive_windows = genome_windows.intersect(merged_chip_bed, u=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)

    # Exclude all windows that overlap a blacklisted region
    blacklist = make_blacklist()
    
    print 'Removing windows that overlap a blacklisted region'
    positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)

    num_positive_windows = positive_windows.count()
    # Binary binding target matrix of all positive windows
    print 'Number of positive windows:', num_positive_windows
    print 'Number of targets:', len(tfs)
    # Generate targets
    print 'Generating target matrix of all positive windows'
    y_positive = parmap.map(intersect_count, chip_beds, positive_windows.fn)
    y_positive = np.array(y_positive, dtype=bool).T
    print 'Positive matrix sparsity', (~y_positive).sum()*1.0/np.prod(y_positive.shape)
    merged_chip_slop_bed = merged_chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
    # Later we want to gather negative windows from the genome that do not overlap
    # with a blacklisted or ChIP region
    nonnegative_regions_bed = merged_chip_slop_bed.cat(blacklist)
    return tfs, positive_windows, y_positive, nonnegative_regions_bed


def load_bigwigs(input_dir):
    bigwig_info_file = input_dir + '/bigwig.txt'
    if not os.path.isfile(bigwig_info_file):
        return [], []
    bigwig_info = np.loadtxt(bigwig_info_file, dtype=str)
    if len(bigwig_info.shape) == 1:
        bigwig_info = np.reshape(bigwig_info, (-1,2))
    bigwig_names = list(bigwig_info[:, 1])
    bigwig_files = [input_dir + '/' + i for i in bigwig_info[:,0]]
    return bigwig_names, bigwig_files


def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output


def make_model(num_tfs, num_bws, num_motifs, num_recurrent, num_dense, dropout_rate):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input
    from keras.layers.convolutional import Convolution1D, MaxPooling1D
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    forward_input = Input(shape=(L, 4 + num_bws,))
    reverse_input = Input(shape=(L, 4 + num_bws,))
    hidden_layers = [
        Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                      filter_length=w, border_mode='valid', activation='relu',
                      subsample_length=1),
        Dropout(0.1),
        TimeDistributed(Dense(num_motifs, activation='relu')),
        MaxPooling1D(pool_length=w2, stride=w2),
        Bidirectional(LSTM(num_recurrent, dropout_W=0.1, dropout_U=0.1, return_sequences=True)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(num_dense, activation='relu'),
        Dense(num_tfs, activation='sigmoid')
    ]
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = merge([forward_output, reverse_output], mode='ave')
    model = Model(input=[forward_input, reverse_input], output=output)

    print 'Compiling model'
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.summary()
    
    return model


def load_model(modeldir):
    tfs_file = modeldir + '/chip.txt'
    tfs = np.loadtxt(tfs_file, dtype=str)
    if len(tfs.shape) == 0:
        tfs = [str(tfs)]
    else:
        tfs = list(tfs)
    bigwig_names_file = modeldir + '/bigwig.txt'
    if not os.path.isfile(bigwig_names_file):
        bigwig_names = []
    else:
        bigwig_names = np.loadtxt(bigwig_names_file, dtype=str)
        if len(bigwig_names.shape) == 0:
            bigwig_names = [str(bigwig_names)]
        else:
            bigwig_names = list(bigwig_names)
    from keras.models import model_from_json
    model_json_file = open(modeldir + '/model.json', 'r')
    model_json = model_json_file.read()
    model = model_from_json(model_json)
    model.load_weights(modeldir + '/best_model.hdf5')
    return tfs, bigwig_names, model


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
    _, _, genome_bed = get_genome_bed()
    genome_bed_test = subset_chroms(test_chroms, genome_bed)
    print 'Windowing test chromosome(s)'
    genome_bed_test_windows = BedTool().window_maker(b=genome_bed_test, w=genome_window_size,
                                                     s=step)
    blacklist = make_blacklist()
    print 'Removing windows overlapping blacklist regions'
    genome_bed_test_windows = genome_bed_test_windows.intersect(blacklist, wa=True, v=True,
                                                                sorted=True)
    print 'Generating labels for test chromosome(s)'
    y = intersect_count(chip_bed, genome_bed_test_windows.fn)
    y = np.array(y, dtype=bool)
    print 'Generating test data iterator'
    bigwig_names, bigwig_files = load_bigwigs(input_dir)
    if positives_only:
        data_test = [(window.chrom, window.start, window.stop, bigwig_files)
                     for y_sample, window in itertools.izip(y, genome_bed_test_windows)
                     if y_sample]
    else:
        data_test = [(window.chrom, window.start, window.stop, bigwig_files)
                     for window in genome_bed_test_windows]
    from iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_test = DataIterator(data_test, genome, 1000, L, bigwig_rc_order, shuffle=False)
    return bigwig_names, datagen_test, y


def load_beddata(tf, genome, bed_file, input_dir):
    bed = BedTool(bed_file)
    blacklist = make_blacklist()
    print 'Determining which windows are valid'
    bed_intersect_blacklist_count = bed.intersect(blacklist, wa=True, c=True, sorted=True)
    nonblacklist_bools = np.array([i.count==0 for i in bed_intersect_blacklist_count])
    print 'Filtering away blacklisted windows'
    bed_filtered = bed.intersect(blacklist, wa=True, v=True, sorted=True)
    print 'Generating test data iterator'
    bigwig_names, bigwig_files = load_bigwigs(input_dir)
    data_bed = [(window.chrom, window.start, window.stop, bigwig_files)
                for window in bed_filtered]
    from iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_bed = DataIterator(data_bed, genome, 1000, L, bigwig_rc_order, shuffle=False)
    return bigwig_names, datagen_bed, nonblacklist_bools
