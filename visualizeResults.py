#!/usr/bin/env python
"""
Script for visualizing results from a trained model.

Use `visualizeResults.py -h` to see an auto-generated description of advanced options.
"""
import numpy as np
from pybedtools import BedTool
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

    conv_layer = Convolution1D(input_dim=4 + num_bws, nb_filter=20,
                               filter_length=26, border_mode='valid', activation='relu',
                               subsample_length=1)
    max_pool_layer = MaxPooling1D(pool_length=13, stride=13)
    dropout_layer = Dropout(0.5)
    flatten_layer = Flatten()
    dense_layer = Dense(975, activation='relu')
    sigmoid_layer = Dense(num_targets, activation='sigmoid')

    forward_conv = conv_layer(forward_input)
    forward_max_pool = max_pool_layer(forward_conv)
    forward_dropout = dropout_layer(forward_max_pool)
    forward_flatten = flatten_layer(forward_dropout)
    forward_dense = dense_layer(forward_flatten)
    forward_sigmoid = sigmoid_layer(forward_dense)

    reverse_conv = conv_layer(reverse_input)
    reverse_max_pool = max_pool_layer(reverse_conv)
    reverse_dropout = dropout_layer(reverse_max_pool)
    reverse_flatten = flatten_layer(reverse_dropout)
    reverse_dense = dense_layer(reverse_flatten)
    reverse_sigmoid = sigmoid_layer(reverse_dense)

    output = merge([forward_sigmoid, reverse_sigmoid], mode='ave')
    model = Model(input=[input_seq, input_bws], output=output)

    print 'Compiling model'
    model.compile('rmsprop', 'binary_crossentropy', metrics=['binary_accuracy'])

    model.summary()

    return model


def output_results(test_seq, test_bws, test_y, seqonly, model_dir, output_dir):
    num_bws = test_bws.shape[-1]
    num_targets = 1#test_y.shape[-1]
    test_y = test_y[:,18]
    num_seqs = len(test_seq)
    if seqonly:
        print '\nTesting only with DNA sequences.'
    model = make_model(num_bws, num_targets, seqonly)
    model.load_weights(model_dir + '/best_model.hdf5')

    if seqonly:
        convlayer = 2
        input = [model.input[0]]
    else:
        convlayer = 6
        input = model.input
    x, y = model.layers[convlayer].get_output_at(0), model.layers[convlayer].get_output_at(1)
    f = theano.function(input,
                        [x.argmax(axis=1), x.max(axis=1), y.argmax(axis=1), y.max(axis=1)])

    print 'Getting activations'
    max_acts = []
    max_inds = []
    max_acts_rc = []
    max_inds_rc = []
    for n in xrange(int(np.ceil(num_seqs*1.0/1000))):
        if seqonly:
            z = f(test_seq[n * 1000:n * 1000 + 1000])
        else:
            z = f(test_seq[n * 1000:n * 1000 + 1000], test_bws[n * 1000:n * 1000 + 1000])
        max_inds += [z[0]]
        max_acts += [z[1]]
        max_inds_rc += [z[2]]
        max_acts_rc += [z[3]]

    max_acts = np.vstack(max_acts)
    max_inds = np.vstack(max_inds)
    max_acts_rc = np.vstack(max_acts_rc)
    max_inds_rc = np.vstack(max_inds_rc)

    test_seq_rc = test_seq[:, ::-1, ::-1]

    print 'Making motifs'

    motifs = np.zeros((20, 26, 4))
    nsites = np.zeros(20)

    for m in xrange(20):
        for n in xrange(num_seqs):
            # Forward strand
            if max_acts[n, m] > 0:
                nsites[m] += 1
                motifs[m] += test_seq[n, max_inds[n, m]:max_inds[n, m]+26, :]
            # Reverse strand
            if max_acts_rc[n, m] > 0:
                nsites[m] += 1
                motifs[m] += test_seq_rc[n, max_inds_rc[n, m]:max_inds_rc[n, m]+26, :]

    f = open(output_dir + '/motifs.txt', 'w')
    f.write('MEME version 4.9.0\n\n'
            'ALPHABET= ACGT\n\n'
            'strands: + -\n\n'
            'Background letter frequencies (from uniform background):\n'
            'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')
    for m in xrange(20):
        f.write('MOTIF M%i O%i\n' % (m, m))
        f.write("letter-probability matrix: alength= 4 w= 26 nsites= %i E= 1337.0e-6\n" % nsites[m])
        for j in xrange(26):
            f.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, :] / np.sum(motifs[m, j, :])))
        f.write('\n')

    f.close()

    test_predicts = model.predict([test_seq, test_bws], batch_size=100)
    roc_auc = np.zeros(num_targets)
    pr_auc = np.zeros(num_targets)

    pylab.rcParams['font.size'] = 20
    pylab.xlabel('FPR')
    pylab.ylabel('TPR')
    pylab.title('ROC curves')
    pylab.plot([0, 1], [0, 1], 'k-', lw=3)
    for t in xrange(num_targets):
        fpr, tpr, _ = roc_curve(test_y, test_predicts)
        roc_auc[t] = auc(fpr, tpr)
        pylab.plot(fpr, tpr, 'k--')
    pylab.show()

    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('PR curves')
    for t in xrange(num_targets):
        precision, recall, _ = precision_recall_curve(test_y,  test_predicts)
        pr_auc[t] = average_precision_score(test_y, test_predicts)
        pylab.plot(recall, precision, 'k--')
    pylab.show()

    np.savetxt(output_dir + '/roc.txt', roc_auc)
    np.savetxt(output_dir + '/pr.txt', pr_auc)


def load_data(input_dir, test_chroms):
    print 'Loading data'

    f = h5py.File(input_dir + '/data.hdf5')
    x_seq = f['x_seq'].value
    x_bws = f['x_bws'].value
    y = f['y'].value
    f.close()
    windows = BedTool(input_dir + '/windows.bed.gz')
    windows_chroms = [window.chrom for window in windows]

    print 'Splitting data'

    # Test
    boolean_list = np.array([chrom in test_chroms for chrom in windows_chroms])
    test_seq = x_seq[boolean_list]
    test_bws = x_bws[boolean_list]
    test_y = y[boolean_list]

    print '\nTest seq shape:', test_seq.shape
    print 'Test bws shape:', test_bws.shape
    print 'Test y shape:', test_y.shape
    print 'Test y sparsity:', 1-test_y.sum()*1.0/np.prod(test_y.shape)

    return test_seq, test_bws, test_y


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
    parser.add_argument('--testchroms', '-t', type=str, required=False, nargs='+',
                        default=['chr10'],
                        help='Chromosome(s) to set aside for testing.')
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
    test_chroms = args.testchroms
    recurrent = args.recurrent
    seqonly = args.seqonly

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

    test_seq, test_bws, test_y = load_data(input_dir, test_chroms)

    output_results(test_seq, test_bws, test_y, seqonly, model_dir, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
