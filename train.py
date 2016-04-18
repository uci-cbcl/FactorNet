#!/usr/bin/env python
"""
Script for training model.

Use `train.py -h` to see an auto-generated description of advanced options.
"""
import numpy as np
from pybedtools import BedTool
import h5py

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input
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


def make_model(num_bws, num_targets):
    print '\nBuilding model'

    conv_layer = Convolution1D(input_dim=4 + num_bws, input_length=1000, nb_filter=320,
                               filter_length=26, border_mode='valid', activation='relu',
                               subsample_length=1)
    max_pool_layer = MaxPooling1D(pool_length=13, stride=13)
    dropout_layer = Dropout(0.5)
    flatten_layer = Flatten()
    dense_layer = Dense(975, activation='relu')
    sigmoid_layer = Dense(num_targets, activation='sigmoid')

    input_seq = Input(shape=(1000, 4,), dtype='bool')
    input_bws = Input(shape=(1000, num_bws,), dtype='float32')
    forward_input = merge([input_seq, input_bws], mode='concat', concat_axis=-2)
    forward_conv = conv_layer(forward_input)
    forward_max_pool = max_pool_layer(forward_conv)
    forward_dropout = dropout_layer(forward_max_pool)
    forward_flatten = flatten_layer(forward_dropout)
    forward_dense = dense_layer(forward_flatten)
    forward_sigmoid = sigmoid_layer(forward_dense)

    input_seq_rc = ReverseComplement()(input_seq)
    input_bws_rc = Reverse()(input_bws)
    reverse_input = merge([input_seq_rc, input_bws_rc], mode='concat', concat_axis=-2)
    reverse_conv = conv_layer(reverse_input)
    reverse_max_pool = max_pool_layer(reverse_conv)
    reverse_dropout = dropout_layer(reverse_max_pool)
    reverse_flatten = flatten_layer(reverse_dropout)
    reverse_dense = dense_layer(reverse_flatten)
    reverse_sigmoid = sigmoid_layer(reverse_dense)

    output = merge([forward_sigmoid, reverse_sigmoid], mode='ave')
    model = Model(input=[input_seq, input_bws], output=output)

    print 'Compiling model'
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

    model.summary()

    return model


def train(train_data, valid_data, test_data, output_dir):
    train_seq, train_bws, train_y = train_data
    valid_seq, valid_bws, valid_y = valid_data
    test_seq, test_bws, test_y = test_data

    num_bws = train_bws.shape[-1]
    num_targets = train_y.shape[-1]
    model = make_model(num_bws, num_targets)

    print 'Running at most 2 epochs'

    checkpointer = ModelCheckpoint(filepath=output_dir + '/best_model.hdf5',
                                   verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    history = model.fit([train_seq, train_bws], train_y,
                        batch_size=100, nb_epoch=2, shuffle=True,
                        validation_data=([valid_seq, valid_bws],
                                         valid_y),
                        callbacks=[checkpointer, earlystopper])

    print 'Saving final model'
    model.save_weights(output_dir + '/final_model.hdf5')

    print 'Saving history'
    history_file = open("train3_history.pkl", 'wb')
    pickle.dump(history.history, history_file)
    history_file.close()

    test_results = model.evaluate([test_seq, test_bws], test_y)

    print 'Test loss:', test_results[0]
    print 'Test accuracy:', test_results[1]


def load_data(input_dir, valid_chroms, test_chroms):
    print 'Loading data'

    f = h5py.File(input_dir + '/data.hdf5')
    x_seq = f['x_seq'].value
    x_bws = f['x_bws'].value
    y = f['y'].value
    f.close()
    windows = BedTool(input_dir + '/windows.bed.gz')
    windows_chroms = [window.chrom for window in windows]

    print 'Splitting data'

    # Valid
    boolean_list = np.array([chrom in valid_chroms for chrom in windows_chroms])
    valid_seq = x_seq[boolean_list]
    valid_bws = x_bws[boolean_list]
    valid_y = y[boolean_list]
    valid_data = (valid_seq, valid_bws, valid_y)

    # Test
    boolean_list = np.array([chrom in test_chroms for chrom in windows_chroms])
    test_seq = x_seq[boolean_list]
    test_bws = x_bws[boolean_list]
    test_y = y[boolean_list]
    test_data = (test_seq, test_bws, test_y)

    # Train
    valid_test_chroms = valid_chroms + test_chroms
    boolean_list = np.array([chrom not in valid_test_chroms for chrom in windows_chroms])
    train_seq = x_seq[boolean_list]
    train_bws = x_bws[boolean_list]
    train_y = y[boolean_list]
    train_data = (train_seq, train_bws, train_y)

    print '\nTrain seq shape:', train_seq.shape
    print 'Train bws shape:', train_bws.shape
    print 'Train y shape:', train_y.shape

    print '\nValid seq shape:', valid_seq.shape
    print 'Valid bws shape:', valid_bws.shape
    print 'Valid y shape:', valid_y.shape

    print '\nTest seq shape:', test_seq.shape
    print 'Test bws shape:', test_bws.shape
    print 'Test y shape:', test_y.shape
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
