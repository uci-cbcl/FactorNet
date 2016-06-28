#!/usr/bin/env python
"""
Script for predicting TF binding with a trained model.

Use `predict.py -h` to see an auto-generated description of advanced options.
"""

import numpy as np
from pybedtools import BedTool
from pybedtools.featurefuncs import extend_fields
from pybedtools.contrib.bigwig import bedgraph_to_bigwig
import pyfasta
import h5py
import theano
import pylab
import matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
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


def make_model(num_bws, num_targets, seqonly):
    print '\nBuilding model'

    input_seq = Input(shape=(1000, 4,))
    input_seq_rc = ReverseComplement()(input_seq)
    input_bws = Input(shape=(1000, num_bws,), dtype='float32')
    input_bws_rc = Reverse()(input_bws)

    if seqonly:
        num_bws = 0
        forward_input = input_seq
        reverse_input = input_seq_rc
    else:
        forward_input = merge([input_seq, input_bws], mode='concat', concat_axis=-1)
        reverse_input = merge([input_seq_rc, input_bws_rc], mode='concat', concat_axis=-1)

    conv_layer = Convolution1D(input_dim=4 + num_bws, nb_filter=120,
                               filter_length=26, border_mode='valid', activation='relu',
                               subsample_length=1)
    max_pool_layer = MaxPooling1D(pool_length=13, stride=13)
    dropout_layer = Dropout(0.2)
    lstm_layer = LSTM(120, return_sequences=True)
    dropout2_layer = Dropout(0.5)
    flatten_layer = Flatten()
    dense_layer = Dense(975, activation='relu')
    sigmoid_layer = Dense(num_targets, activation='sigmoid')

    forward_conv = conv_layer(forward_input)
    forward_max_pool = max_pool_layer(forward_conv)
    forward_dropout = dropout_layer(forward_max_pool)
    forward_lstm = lstm_layer(forward_dropout)
    forward_dropout2 = dropout2_layer(forward_lstm)
    forward_flatten = flatten_layer(forward_dropout2)
    forward_dense = dense_layer(forward_flatten)
    forward_sigmoid = sigmoid_layer(forward_dense)

    reverse_conv = conv_layer(reverse_input)
    reverse_max_pool = max_pool_layer(reverse_conv)
    reverse_dropout = dropout_layer(reverse_max_pool)
    reverse_lstm = lstm_layer(reverse_dropout)
    reverse_dropout2 = dropout2_layer(reverse_lstm)
    reverse_flatten = flatten_layer(reverse_dropout2)
    reverse_dense = dense_layer(reverse_flatten)
    reverse_sigmoid = sigmoid_layer(reverse_dense)

    output = merge([forward_sigmoid, reverse_sigmoid], mode='ave')
    model = Model(input=[input_seq, input_bws], output=output)

    print 'Compiling model'
    model.compile('rmsprop', 'binary_crossentropy', metrics=['binary_accuracy'])

    model.summary()

    return model


def add_score(feature, i, scores):
    feature[3] = str(scores[i[0]])
    i[0] += 1
    return feature


def output_results(x_seq, x_bws, y, windows, tfs, seqonly, model_dir, output_dir):
    num_bws = x_bws.shape[-1]
    num_targets = len(tfs)
    num_seqs = len(x_seq)
    if seqonly:
        print '\nTesting only with DNA sequences.'
    model = make_model(num_bws, num_targets, seqonly)
    model.load_weights(model_dir + '/best_model.hdf5')
    predicts = model.predict([x_seq, x_bws], batch_size=1000, verbose=1)
    windows2 = windows.each(extend_fields, 4).saveas()
    for j in xrange(len(tfs)):
        tf = tfs[j]
        print tf
        scores = predicts[:, j]
        i = [0]
        windows3 = windows2.each(add_score, i, scores).saveas()
        bedgraph_to_bigwig(windows3, 'hg19', output_dir + '/' + tf + '.bw')


def load_data(input_dir):
    print 'Loading data'

    f = h5py.File(input_dir + '/data.hdf5')
    x_seq = f['x_seq'].value
    x_bws = f['x_bws'].value
    y = f['y'].value
    f.close()
    windows = BedTool(input_dir + '/windows.bed.gz')

    print '\nseq shape:', x_seq.shape
    print 'bws shape:', x_bws.shape
    print 'y shape:', y.shape
    print 'y sparsity:', 1-y.sum()*1.0/np.prod(y.shape)

    return x_seq, x_bws, y, windows


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Visualize results of a trained model.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--inputdir', '-i', type=str, required=True,
                        help='Folder containing data generated by makeData.py')
    parser.add_argument('--modeldir', '-m', type=str, required=True,
                        help='Folder containing trained model generated by train.py.')
    parser.add_argument('--tfs', '-t', type=str, required=True,
                        help='File containing list of TFs.')
    parser.add_argument('--recurrent', '-r', action='store_true',
                        help='Save data as long sequences for fully recurrent models.')
    parser.add_argument('--seqonly', '-s', action='store_true',
                        help='Train only with sequences.')
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
    model_dir = args.modeldir
    recurrent = args.recurrent
    seqonly = args.seqonly
    tfs = np.loadtxt(args.tfs, dtype=str)

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

    x_seq, x_bws, test_y, windows = load_data(input_dir)

    output_results(x_seq, x_bws, test_y, windows, tfs, seqonly, model_dir, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()