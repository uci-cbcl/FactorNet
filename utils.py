import numpy as np
from pybedtools import BedTool, Interval
import pyfasta
import parmap

# Standard library imports
import os
import itertools

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


def make_features_multiTask(positive_windows, y_positive, nonnegative_regions_bed, 
                            bigwig_files, bigwig_names, genome, epochs, valid_chroms, test_chroms):
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

    # Train
    print 'Extracting data from negative training BEDs'
    negative_targets = np.zeros(y_positive.shape[1])
    negative_data_train = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_train]

    # Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_valid]
    
    num_positive_train_windows = len(positive_data_train)
    
    data_valid = negative_data_valid + positive_data_valid

    print 'Shuffling training data'
    data_train = []
    for i in xrange(epochs):
        epoch_data = []
        epoch_data.extend(positive_data_train)
        epoch_data.extend(negative_data_train[i*num_positive_train_windows:(i+1)*num_positive_train_windows])
        np.random.shuffle(epoch_data)
        data_train.extend(epoch_data)

    print 'Generating data iterators'
    from data_iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order)
    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order)

    print len(datagen_train), 'training samples'
    print len(datagen_valid), 'validation samples'
    return datagen_train, datagen_valid


def data_to_bed(data):
    intervals = []
    for datum in data:
        chrom = datum[0]
        start = datum[1]
        stop = datum[2]
        intervals.append(Interval(chrom, start, stop))
    return BedTool(intervals)


def extract_data_from_bed(args, shift, label, gencode):
    peaks = args[0]
    bigwig_files = args[1]
    meta = args[2]

    data = []    

    if gencode:
        cpg_bed = BedTool('resources/cpgisland.bed.gz')
        cds_bed = BedTool('resources/wgEncodeGencodeBasicV19.cds.merged.bed.gz')
        intron_bed = BedTool('resources/wgEncodeGencodeBasicV19.intron.merged.bed.gz')
        promoter_bed = BedTool('resources/wgEncodeGencodeBasicV19.promoter.merged.bed.gz')
        utr5_bed = BedTool('resources/wgEncodeGencodeBasicV19.utr5.merged.bed.gz')
        utr3_bed = BedTool('resources/wgEncodeGencodeBasicV19.utr3.merged.bed.gz')

        peaks_cpg_bedgraph = peaks.intersect(cpg_bed, wa=True, c=True)
        peaks_cds_bedgraph = peaks.intersect(cds_bed, wa=True, c=True)
        peaks_intron_bedgraph = peaks.intersect(intron_bed, wa=True, c=True)
        peaks_promoter_bedgraph = peaks.intersect(promoter_bed, wa=True, c=True)
        peaks_utr5_bedgraph = peaks.intersect(utr5_bed, wa=True, c=True)
        peaks_utr3_bedgraph = peaks.intersect(utr3_bed, wa=True, c=True)

        for cpg, cds, intron, promoter, utr5, utr3 in itertools.izip(peaks_cpg_bedgraph,peaks_cds_bedgraph,peaks_intron_bedgraph,peaks_promoter_bedgraph,peaks_utr5_bedgraph,peaks_utr3_bedgraph):
            chrom = cpg.chrom
            peak_start = cpg.start
            peak_stop = cpg.stop
            peak_mid = (peak_start + peak_stop)/2
            start = peak_mid - genome_window_size/2
            stop = peak_mid + genome_window_size/2
            if shift:
                shift_size = peak_stop - start - 75 - 1
            else:
                shift_size = 0
            gencode = np.array([cpg.count, cds.count, intron.count, promoter.count, utr5.count, utr3.count], dtype=bool)
            meta_gencode = np.append(meta, gencode)
            data.append((chrom, start, stop, shift_size, bigwig_files, meta_gencode, label))
    else:
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
    randomseed = args[3]
    if num_copies > 1:
        positive_windows = BedTool.cat(*(num_copies * [positive_windows]), postmerge=False)
    negative_windows = positive_windows.shuffle(g=genome_sizes_file,
                                                incl=include_bed.fn,
                                                excl=nonnegative_regions_bed.fn,
                                                noOverlapping=noOverlapping,
                                                seed=randomseed)
    return negative_windows


def make_features_singleTask(chip_bed_list, nonnegative_regions_bed_list, bigwig_files_list, bigwig_names,
                             meta_list, gencode, genome, epochs, negatives, valid_chroms, test_chroms, 
                             valid_chip_bed_list, valid_nonnegative_regions_bed_list,
                             valid_bigwig_files_list, valid_meta_list):
    num_cells = len(chip_bed_list)
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    genome_bed_train, genome_bed_valid, genome_bed_test = \
        [subset_chroms(chroms_set, genome_bed) for chroms_set in
         (train_chroms, valid_chroms, test_chroms)]

    print 'Splitting ChIP peaks into training, validation, and testing BEDs'
    chip_bed_split_list = parmap.map(valid_test_split_wrapper, chip_bed_list, valid_chroms, test_chroms)
    chip_bed_train_list, chip_bed_valid_list, chip_bed_test_list = zip(*chip_bed_split_list)

    if valid_chip_bed_list: # the user specified a validation directory, must adjust validation data
        valid_chip_bed_split_list = parmap.map(valid_test_split_wrapper, valid_chip_bed_list, valid_chroms, test_chroms)
        _, chip_bed_valid_list, _ = zip(*valid_chip_bed_split_list)
    else:
        valid_nonnegative_regions_bed_list = nonnegative_regions_bed_list
        valid_bigwig_files_list = bigwig_files_list
        valid_meta_list = meta_list

    positive_label = [True]
    #Train
    print 'Extracting data from positive training BEDs'
    positive_data_train_list = parmap.map(extract_data_from_bed,
                                          zip(chip_bed_train_list, bigwig_files_list, meta_list),
                                          True, positive_label, gencode)
    positive_data_train = list(itertools.chain(*positive_data_train_list))

    #Validation
    print 'Extracting data from positive validation BEDs'
    positive_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(chip_bed_valid_list, valid_bigwig_files_list, valid_meta_list),
                                          False, positive_label, gencode)
    positive_data_valid = list(itertools.chain(*positive_data_valid_list))

    print 'Shuffling positive training windows in negative regions'
    train_noOverlap = True
    train_randomseeds = np.random.randint(-214783648, 2147483647, num_cells)
    positive_windows_train_list = parmap.map(data_to_bed, positive_data_train_list)
    negative_windows_train_list = parmap.map(negative_shuffle_wrapper,
                                             zip(positive_windows_train_list, nonnegative_regions_bed_list,
                                                 bigwig_files_list, train_randomseeds),
                                             genome_bed_train, negatives*epochs, train_noOverlap)

    print 'Shuffling positive validation windows in negative regions'
    valid_randomseeds = np.random.randint(-214783648, 2147483647, num_cells)
    positive_windows_valid_list = parmap.map(data_to_bed, positive_data_valid_list)
    negative_windows_valid_list = parmap.map(negative_shuffle_wrapper,
                                             zip(positive_windows_valid_list, nonnegative_regions_bed_list,
                                                 bigwig_files_list, valid_randomseeds),
                                             genome_bed_valid, negatives, True)

    negative_label = [False]
    #Train
    print 'Extracting data from negative training BEDs'
    negative_data_train_list = parmap.map(extract_data_from_bed,
                                          zip(negative_windows_train_list, bigwig_files_list, meta_list),
                                          False, negative_label, gencode)
    negative_data_train = list(itertools.chain(*negative_data_train_list))

    #Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid_list = parmap.map(extract_data_from_bed,
                                          zip(negative_windows_valid_list, valid_bigwig_files_list, valid_meta_list),
                                          False, negative_label, gencode)
    negative_data_valid = list(itertools.chain(*negative_data_valid_list))

    data_valid = negative_data_valid + positive_data_valid

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
    from data_iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order)
    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order, shuffle=True)

    print len(datagen_train), 'training samples'
    print len(datagen_valid), 'validation samples'
    return datagen_train, datagen_valid


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


def load_chip_multiTask(input_dir):
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
    chroms, chroms_sizes, genome_bed = get_genome_bed()
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


def load_chip_singleTask(input_dirs, tf):
    blacklist = make_blacklist()

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
    from keras.layers.pooling import GlobalMaxPooling1D
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    forward_input = Input(shape=(L, 4 + num_bws,))
    reverse_input = Input(shape=(L, 4 + num_bws,))
    if num_recurrent < 0:
        hidden_layers = [
            Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                          filter_length=w, border_mode='valid', activation='relu',
                          subsample_length=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_motifs, activation='relu')),
            GlobalMaxPooling1D(),
            Dropout(dropout_rate),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_tfs, activation='sigmoid')
        ]
    elif num_recurrent == 0:
        hidden_layers = [
            Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                          filter_length=w, border_mode='valid', activation='relu',
                          subsample_length=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_motifs, activation='relu')),
            MaxPooling1D(pool_length=w2, stride=w2),
            Dropout(dropout_rate),
            Flatten(),
            Dense(num_dense, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_tfs, activation='sigmoid')
        ]
    else:
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
            Dropout(dropout_rate),
            Dense(num_tfs, activation='sigmoid')
        ]
    forward_output = get_output(forward_input, hidden_layers)     
    reverse_output = get_output(reverse_input, hidden_layers)
    output = merge([forward_output, reverse_output], mode='ave')
    model = Model(input=[forward_input, reverse_input], output=output)

    return model


def make_meta_model(num_tfs, num_bws, num_meta, num_motifs, num_recurrent, num_dense, dropout_rate):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input
    from keras.layers.convolutional import Convolution1D, MaxPooling1D
    from keras.layers.pooling import GlobalMaxPooling1D
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    forward_input = Input(shape=(L, 4 + num_bws,))
    reverse_input = Input(shape=(L, 4 + num_bws,))
    meta_input = Input(shape=(num_meta,))
    if num_recurrent < 0:
        hidden_layers = [
            Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                          filter_length=w, border_mode='valid', activation='relu',
                          subsample_length=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_motifs, activation='relu')),
            GlobalMaxPooling1D(),
            Dropout(dropout_rate),
            Dense(num_dense, activation='relu'),
        ]
    elif num_recurrent == 0:
        hidden_layers = [
            Convolution1D(input_dim=4 + num_bws, nb_filter=num_motifs,
                          filter_length=w, border_mode='valid', activation='relu',
                          subsample_length=1),
            Dropout(0.1),
            TimeDistributed(Dense(num_motifs, activation='relu')),
            MaxPooling1D(pool_length=w2, stride=w2),
            Dropout(dropout_rate),
            Flatten(),
            Dense(num_dense, activation='relu'),
        ]
    else:
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
        ]
    forward_dense = get_output(forward_input, hidden_layers)     
    reverse_dense = get_output(reverse_input, hidden_layers)
    forward_dense = merge([forward_dense, meta_input], mode='concat')
    reverse_dense = merge([reverse_dense, meta_input], mode='concat')
    dropout2_layer = Dropout(0.1)
    forward_dropout2 = dropout2_layer(forward_dense)
    reverse_dropout2 = dropout2_layer(reverse_dense)
    dense2_layer = Dense(num_dense, activation='relu')
    forward_dense2 = dense2_layer(forward_dropout2)
    reverse_dense2 = dense2_layer(reverse_dropout2)
    dropout3_layer = Dropout(dropout_rate)
    forward_dropout3 = dropout3_layer(forward_dense2)
    reverse_dropout3 = dropout3_layer(reverse_dense2)
    sigmoid_layer =  Dense(num_tfs, activation='sigmoid')
    forward_output = sigmoid_layer(forward_dropout3)
    reverse_output = sigmoid_layer(reverse_dropout3)
    output = merge([forward_output, reverse_output], mode='ave')
    model = Model(input=[forward_input, reverse_input, meta_input], output=output)

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
    features_file = modeldir + '/feature.txt'
    assert os.path.isfile(features_file)
    features = np.loadtxt(features_file, dtype=str)
    if len(features.shape) == 0:
        features = [str(features)]
    else:
        features = list(features)
    from keras.models import model_from_json
    model_json_file = open(modeldir + '/model.json', 'r')
    model_json = model_json_file.read()
    model = model_from_json(model_json)
    model.load_weights(modeldir + '/best_model.hdf5')
    return tfs, bigwig_names, features, model


def load_motif_db(meme_filename):
    f = open(meme_filename, 'r')
    lines = f.readlines()    
    f.close()
    num_lines = len(lines)
    i = 0
    motifs_dict = {}
    while i < num_lines:
        line = lines[i]
        if 'MOTIF' in line:
            motif_name = line.split()[-1].strip().upper()
            while 'letter-probability matrix' not in line:
                i += 1
                line = lines[i]
            motif_info = lines[i]
            motif_info = motif_info.split()
            w_index = motif_info.index('w=') + 1
            w = int(motif_info[w_index])
            motif = np.zeros((w,4))
            i += 1
            line = lines[i]
            while len(line.strip()) == 0:
                i += 1
                line = lines[i]
            for j in xrange(w):
                motif[j,:] = np.array(lines[i].split(), dtype=float)
                i += 1
            motifs_dict[motif_name] = motif
        i += 1
    return motifs_dict


def inject_pwm(model, pwm):
    layer_names = [l.name for l in model.layers]
    conv_layer_index = layer_names.index('convolution1d_1')
    conv_layer = model.layers[conv_layer_index]
    w = conv_layer.filter_length
    pwm_length = len(pwm)
    pwm_start = w/2 - pwm_length/2
    pwm_stop = pwm_start + pwm_length
    pwm_r = pwm[::-1, ::-1]

    conv_weights = conv_layer.get_weights()

    conv_weights[0][:, :, :, 0] = 0
    conv_weights[0][pwm_start:pwm_stop, 0, 0:4, 0] = pwm[::-1]
    conv_weights[1][0] = 0
    conv_weights[0][:, :, :, 1] = 0
    conv_weights[0][pwm_start:pwm_stop, 0, 0:4, 1] = pwm_r[::-1]
    conv_weights[1][1] = 0

    conv_layer.set_weights(conv_weights)


def load_beddata(genome, bed_file, use_meta, use_gencode, input_dir, is_sorted, chrom=None):
    bed = BedTool(bed_file)
    if not is_sorted:
        print 'Sorting BED file'
        bed = bed.sort()
        is_sorted = True
    blacklist = make_blacklist()
    print 'Determining which windows are valid'
    bed_intersect_blacklist_count = bed.intersect(blacklist, wa=True, c=True, sorted=is_sorted)
    if chrom:
        nonblacklist_bools = np.array([i.chrom==chrom and i.count==0 for i in bed_intersect_blacklist_count])
    else:
        nonblacklist_bools = np.array([i.count==0 for i in bed_intersect_blacklist_count])
    print 'Filtering away blacklisted windows'
    bed_filtered = bed.intersect(blacklist, wa=True, v=True, sorted=is_sorted)
    if chrom:
        print 'Filtering away windows not in chromosome:', chrom
        bed_filtered = subset_chroms([chrom], bed_filtered)
    print 'Generating test data iterator'
    bigwig_names, bigwig_files_list = load_bigwigs([input_dir])
    bigwig_files = bigwig_files_list[0]
    if use_meta:
        meta_names, meta_list = load_meta([input_dir])
        meta = meta_list[0]
    else:
        meta = []
        meta_names = None
    
    shift = 0
    
    if use_gencode:
        cpg_bed = BedTool('resources/cpgisland.bed.gz')
        cds_bed = BedTool('resources/wgEncodeGencodeBasicV19.cds.merged.bed.gz')
        intron_bed = BedTool('resources/wgEncodeGencodeBasicV19.intron.merged.bed.gz')
        promoter_bed = BedTool('resources/wgEncodeGencodeBasicV19.promoter.merged.bed.gz')
        utr5_bed = BedTool('resources/wgEncodeGencodeBasicV19.utr5.merged.bed.gz')
        utr3_bed = BedTool('resources/wgEncodeGencodeBasicV19.utr3.merged.bed.gz')

        peaks_cpg_bedgraph = bed_filtered.intersect(cpg_bed, wa=True, c=True)
        peaks_cds_bedgraph = bed_filtered.intersect(cds_bed, wa=True, c=True)
        peaks_intron_bedgraph = bed_filtered.intersect(intron_bed, wa=True, c=True)
        peaks_promoter_bedgraph = bed_filtered.intersect(promoter_bed, wa=True, c=True)
        peaks_utr5_bedgraph = bed_filtered.intersect(utr5_bed, wa=True, c=True)
        peaks_utr3_bedgraph = bed_filtered.intersect(utr3_bed, wa=True, c=True)

        data_bed = [(window.chrom, window.start, window.stop, 0, bigwig_files, np.append(meta, np.array([cpg.count, cds.count, intron.count, promoter.count, utr5.count, utr3.count], dtype=bool)))
                    for window, cpg, cds, intron, promoter, utr5, utr3 in 
                    itertools.izip(bed_filtered, peaks_cpg_bedgraph,peaks_cds_bedgraph,peaks_intron_bedgraph,peaks_promoter_bedgraph,peaks_utr5_bedgraph,peaks_utr3_bedgraph)]
    else:
        data_bed = [(window.chrom, window.start, window.stop, shift, bigwig_files, meta)
                    for window in bed_filtered]
    from data_iter import DataIterator
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_bed = DataIterator(data_bed, genome, 1000, L, bigwig_rc_order, shuffle=False)
    return bigwig_names, meta_names, datagen_bed, nonblacklist_bools
