#!/usr/bin/env python
"""
Script for training model.

Use `train.py -h` to see an auto-generated description of advanced options.
"""
import utils

# Standard library imports
import sys
import os
import errno
import argparse
import pickle


def train(datagen_train, datagen_valid, datagen_test, model, epochs, output_dir):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    print 'Running at most', str(epochs), 'epochs'

    checkpointer = ModelCheckpoint(filepath=output_dir + '/best_model.hdf5',
                                   verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=epochs, verbose=1)

    train_samples_per_epoch = len(datagen_train)/epochs/utils.batch_size*utils.batch_size
    history = model.fit_generator(datagen_train, samples_per_epoch=train_samples_per_epoch,
                                  nb_epoch=epochs, validation_data=datagen_valid,
                                  nb_val_samples=len(datagen_valid),
                                  callbacks=[checkpointer, earlystopper],
                                  pickle_safe=True)

    print 'Saving final model'
    model.save_weights(output_dir + '/final_model.hdf5', overwrite=True)

    print 'Saving history'
    history_file = open(output_dir + '/history.pkl', 'wb')
    pickle.dump(history.history, history_file)
    history_file.close()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Train model.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--inputdir', '-i', type=str, required=True,
                        help='Folder containing data generated by makeData.py')
    parser.add_argument('--validchroms', '-v', type=str, required=False, nargs='+',
                        default=['chr11'],
                        help='Chromosome(s) to set aside for validation.')
    parser.add_argument('--testchroms', '-t', type=str, required=False, nargs='+',
                        default=['chr1', 'chr8', 'chr21'],
                        help='Chromosome(s) to set aside for testing.')
    parser.add_argument('--epochs', '-e', type=int, required=False,
                        default=20,
                        help='Epochs to train (default: 20).')
    parser.add_argument('--kernels', '-k', type=int, required=False,
                        default=128,
                        help='Number of kernels in model (default: 128).')
    parser.add_argument('--recurrent', '-r', type=int, required=False,
                        default=128,
                        help='Number of LSTM units in model (default: 128).')
    parser.add_argument('--dense', '-d', type=int, required=False,
                        default=256,
                        help='Number of dense units in model (default: 256).')
    parser.add_argument('--seed', '-s', type=int, required=False,
                        default=420,
                        help='Random seed for consistency (default: 420).')
    parser.add_argument('--dropout', '-p', type=float, required=False,
                        default=0.5,
                        help='Dropout rate between the LSTM and dense layers (default: 0.5).')
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
    epochs = args.epochs
    seed = args.seed
    utils.set_seed(seed)
    dropout_rate = args.dropout

    num_motifs = args.kernels
    num_recurrent = args.recurrent
    num_dense = args.dense

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

    print 'loading genome'
    genome = utils.load_genome()
    print 'loading ChIP labels'
    tfs, positive_windows, y_positive, nonnegative_regions_bed = utils.load_chip(input_dir)
    print 'loading bigWig data'
    bigwig_names, bigwig_files = utils.load_bigwigs(input_dir)
    print 'making features'
    datagen_train, datagen_valid, datagen_test = utils.make_features(positive_windows, y_positive,
                                                                     nonnegative_regions_bed,
                                                                     bigwig_files, bigwig_names,
                                                                     genome, epochs,
                                                                     valid_chroms, test_chroms)
    print 'building model'
    model = utils.make_model(len(tfs), len(bigwig_names), num_motifs, num_recurrent, num_dense, dropout_rate)

    output_tf_file = open(output_dir + '/chip.txt', 'w')
    for tf in tfs:
        output_tf_file.write("%s\n" % tf)
    output_tf_file.close()
    output_bw_file = open(output_dir + '/bigwig.txt', 'w')
    for bw in bigwig_names:
        output_bw_file.write("%s\n" % bw)
    output_bw_file.close()
    model_json = model.to_json()
    output_json_file = open(output_dir + '/model.json', 'w')
    output_json_file.write(model_json)
    output_json_file.close()
    train(datagen_train, datagen_valid, datagen_test, model, epochs, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
