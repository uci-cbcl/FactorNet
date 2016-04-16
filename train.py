#!/usr/bin/env python
import numpy as np
from pybedtools import BedTool
import h5py

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

# Standard library imports
import sys
import os
import errno
import argparse


class Reverse(Layer):

    def call(self, x, mask):
        return x[:, ::-1, :]


class ReverseComplement(Layer):

    def call(self, x, mask):
        return x[:, ::-1, ::-1]


def make_model(num_bws):
    print 'Building model'

    model = Model(input=[input_seq, input_bws], output=output)
    """
    forward_seq = Sequential()
    forward_seq.add(Identity(input_shape=(1000, 4,)))
    forward_bws = Sequential()
    forward_seq.add(Identity(input_shape=(1000, num_bws,)))
    forward = Sequential()
    forward_inputs = [forward_seq, forward_bws]
    forward.add(Merge(forward_inputs, mode='concat', concat_axis=1))

    reverse_seq = Sequential()
    reverse_seq.add(Identity(input_shape=(1000, 4,)))
    reverse_bws = Sequential()
    reverse_seq.add(Identity(input_shape=(1000, num_bws,)))
    reverse = Sequential()
    reverse_inputs = [reverse_seq, reverse_bws]
    reverse.add(Merge(reverse_inputs, mode='concat', concat_axis=1))

    model = Sequential()

    inputs = [forward, reverse]

    add_shared_layer(Convolution1D(input_dim=4 + num_bws, input_length=1000, nb_filter=320,
                                   filter_length=26, border_mode="valid", activation="relu",
                                   subsample_length=1), inputs)

    add_shared_layer(MaxPooling1D(pool_length=13, stride=13), inputs)

    add_shared_layer(Dropout(0.5), inputs)

    add_shared_layer(Flatten(), inputs)

    add_shared_layer(Dense(input_dim=320 * 75, output_dim=925, activation="relu"), inputs)

    add_shared_layer(Dense(input_dim=925, output_dim=1, activation="sigmoid"), inputs)

    model.add(Merge(inputs, mode='ave'))
    """

    print 'Compiling model'
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

    model.summary()

    return model


def train(train_data, valid_data, test_data, output_dir):
    train_seq, train_bws, train_y = train_data
    train_seq_rc = train_seq[:, ::-1, ::-1]
    train_bws_rc = train_bws[:, ::-1, :]

    valid_seq, valid_bws, valid_y = valid_data
    valid_seq_rc = valid_seq[:, ::-1, ::-1]
    valid_bws_rc = valid_bws[:, ::-1, :]

    test_seq, test_bws, test_y = test_data
    test_seq_rc = test_seq[:, ::-1, ::-1]
    test_bws_rc = test_bws[:, ::-1, :]

    model = make_model()

    print 'Running at most 30 epochs'

    checkpointer = ModelCheckpoint(filepath=output_dir + '/best_model.hdf5',
                                   verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

    history = model.fit([train_seq, train_bws, train_seq_rc, train_bws_rc], train_y,
                        batch_size=100, nb_epoch=30, shuffle=True,
                        validation_data=([valid_seq, valid_bws, valid_seq_rc, valid_bws_rc],
                                         valid_y),
                        callbacks=[checkpointer, earlystopper])

    print 'Saving final model'
    model.save_weights(output_dir + '/final_model.hdf5')

    print 'Saving history'
    history_file = open("train3_history.pkl", 'wb')
    pickle.dump(history.history, history_file)
    history_file.close()

    test_results = model.evaluate([test_seq, test_bws, test_seq_rc, test_bws_rc], test_y)

    print 'Test loss:', test_results[0]
    print 'Test accuracy:', test_results[1]


def load_data(input_dir, valid_chroms, test_chroms):
    print 'Loading data'

    f = h5py.File(input_dir + '/data.hdf5')
    windows = BedTool(input_dir + '/windows.bed.gz')
    windows_chroms = [window.chrom for window in windows]

    # Valid
    boolean_list = [chrom in valid_chroms for chrom in windows_chroms]
    valid_seq = f['x_seq'][boolean_list]
    valid_bws = f['x_bws'][boolean_list]
    valid_y = f['y'][boolean_list]
    valid_data = (valid_seq, valid_bws, valid_y)

    # Test
    boolean_list = [chrom in test_chroms for chrom in windows_chroms]
    test_seq = f['x_seq'][boolean_list]
    test_bws = f['x_bws'][boolean_list]
    test_y = f['y'][boolean_list]
    test_data = (test_seq, test_bws, test_y)

    # Train
    valid_test_chroms = valid_chroms + test_chroms
    boolean_list = [chrom not in valid_test_chroms for chrom in windows_chroms]
    train_seq = f['x_seq'][boolean_list]
    train_bws = f['x_bws'][boolean_list]
    train_y = f['y'][boolean_list]
    train_data = (train_seq, train_bws, train_y)
    f.close()

    return train_data, valid_data, test_data


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Generate training, validation, and testing sets.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--inputdir', '-i', type=str, required=True,
                        help='Folder containing data generated by makeData.py')
    parser.add_argument('--validchroms', '-v', type=str, required=False, nargs='+',
                        default=['chr11'],
                        help='Chromosome(s) to set aside for validation.')
    parser.add_argument('--testchroms', '-t', type=str, required=False, nargs='+',
                        default=['chr10'],
                        help='Chromosome(s) to set aside for testing.')
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

    input_dir = args.inputdir
    valid_chroms = args.validchroms
    test_chroms = args.testchroms
    recurrent = args.recurrent

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

    train_data, valid_data, test_data = load_data(input_dir, valid_chroms, test_chroms)

    train(train_data, valid_data, test_data, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
