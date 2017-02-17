#!/usr/bin/env python
"""
Script for visualizing results from a trained model. Kernels from the first convolutional layer are converted to histogram plots and a .meme file. 

Use `visualize.py -h` to see an auto-generated description of advanced options.
"""
import numpy as np
import utils

# Standard library imports
import sys
import os
import errno
import argparse


def output_results(bigwig_names, datagen_bed, model, output_dir):
    from keras import backend as K
    from keras.models import Model
    print 'Visualizing first convolutional layer'
    layer_names = [l.name for l in model.layers]
    conv_layer_index = layer_names.index('convolution1d_1')
    conv_layer = model.layers[conv_layer_index]
    num_motifs = conv_layer.nb_filter
    w = conv_layer.filter_length

    fwd_conv, rev_conv = conv_layer.get_output_at(0), conv_layer.get_output_at(1)
    f = K.function(model.input,
                   [fwd_conv.argmax(axis=1), fwd_conv.max(axis=1),
                    rev_conv.argmax(axis=1), rev_conv.max(axis=1)])

    print 'Getting activations'

    num_seqs = len(datagen_bed)
    print 'Processing', str(num_seqs), 'sequences'

    motifs = np.zeros((num_motifs, w, 4 + len(bigwig_names)))
    nsites = np.zeros(num_motifs)

    num_seqs_processed = 0
    while num_seqs_processed < num_seqs:
        x = datagen_bed.next()
        num_seqs_processed += len(x[0])
        z = f(x)
        max_inds = z[0] # N x M matrix, where M is the number of motifs
        max_acts = z[1]
        max_inds_rc = z[2]
        max_acts_rc = z[3]
        for m in xrange(num_motifs):
            for n in xrange(len(x[0])):
                # Forward strand
                if max_acts[n, m] > 0:
                    nsites[m] += 1
                    motifs[m] += x[0][n, max_inds[n, m]:max_inds[n, m] + w, :]
                # Reverse strand
                if max_acts_rc[n, m] > 0:
                    nsites[m] += 1
                    motifs[m] += x[1][n, max_inds_rc[n, m]:max_inds_rc[n, m] + w, :]

    print 'Making motifs'

    f = open(output_dir + '/motifs.txt', 'w')
    f.write('MEME version 4.9.0\n\n'
            'ALPHABET= ACGT\n\n'
            'strands: + -\n\n'
            'Background letter frequencies (from uniform background):\n'
            'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')
    for m in xrange(num_motifs):
        if nsites[m] == 0:
            continue
        f.write('MOTIF M%i O%i\n' % (m, m))
        f.write("letter-probability matrix: alength= 4 w= 26 nsites= %i E= 1337.0e-6\n" % nsites[m])
        for j in xrange(w):
            f.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
        f.write('\n')

    f.close()

    print 'Saving mean bigwig signals'
    motifs_bws = motifs[:, :, 4:]
    for m in xrange(num_motifs):
        motifs_bws[m] = motifs_bws[m] / nsites[m]

    for i, bigwig_name in enumerate(bigwig_names):
        np.save(output_dir + '/' + bigwig_name + '_meansignal.npy', motifs_bws[:, :, i])

    print 'Visualizing time distributed layer'
    datagen_bed.reset()
    timedistributed_layer_index = layer_names.index('timedistributed_1')
    timedistributed_layer = model.layers[timedistributed_layer_index]

    fwd_timedistributed, rev_timedistributed = timedistributed_layer.get_output_at(0), timedistributed_layer.get_output_at(1)
    intermediate_layer_model = Model(input=model.input,output=[fwd_timedistributed, rev_timedistributed])

    print 'Getting activations'

    motifs = np.zeros((num_motifs, w, 4 + len(bigwig_names)))
    nsites = np.zeros(num_motifs)

    num_seqs_processed = 0
    while num_seqs_processed < num_seqs:
        x = datagen_bed.next()
        num_seqs_processed += len(x[0])
        z = intermediate_layer_model.predict(x)
        max_inds = z[0].argmax(axis=1) # N x M matrix, where M is the number of motifs
        max_acts = z[0].max(axis=1)
        max_inds_rc = z[1].argmax(axis=1)
        max_acts_rc = z[1].max(axis=1)
        for m in xrange(num_motifs):
            for n in xrange(len(x[0])):
                # Forward strand
                if max_acts[n, m] > 0:
                    nsites[m] += 1
                    motifs[m] += x[0][n, max_inds[n, m]:max_inds[n, m] + w, :]
                # Reverse strand
                if max_acts_rc[n, m] > 0:
                    nsites[m] += 1
                    motifs[m] += x[1][n, max_inds_rc[n, m]:max_inds_rc[n, m] + w, :]

    print 'Making motifs'

    f = open(output_dir + '/motifs2.txt', 'w')
    f.write('MEME version 4.9.0\n\n'
            'ALPHABET= ACGT\n\n'
            'strands: + -\n\n'
            'Background letter frequencies (from uniform background):\n'
            'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')
    for m in xrange(num_motifs):
        if nsites[m] == 0:
            continue
        f.write('MOTIF M%i O%i\n' % (m, m))
        f.write("letter-probability matrix: alength= 4 w= 26 nsites= %i E= 1337.0e-6\n" % nsites[m])
        for j in xrange(w):
            f.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
        f.write('\n')

    f.close()

    print 'Saving mean bigwig signals'
    motifs_bws = motifs[:, :, 4:]
    for m in xrange(num_motifs):
        motifs_bws[m] = motifs_bws[m] / nsites[m]

    for i, bigwig_name in enumerate(bigwig_names):
        np.save(output_dir + '/' + bigwig_name + '_meansignal2.npy', motifs_bws[:, :, i])


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
                        help='Folder containing data.')
    parser.add_argument('--modeldir', '-m', type=str, required=True,
                        help='Folder containing trained model generated by train.py.')
    parser.add_argument('--bed', '-b', type=str, required=True,
                        help='BED file containing intervals to generate predictions from. Typically a ChIP or DNase peak file.')
    parser.add_argument('--chrom', '-c', type=str, required=False,
                        default='chr11',
                        help='Chromosome to use for visualization. Only sequences on this chromosome will be used.')
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
    bed_file = args.bed
    chrom = args.chrom

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

    print 'Loading genome'
    genome = utils.load_genome()
    print 'Loading model'
    model_tfs, model_bigwig_names, features, model = utils.load_model(model_dir)
    use_meta = 'meta' in features
    use_gencode = 'gencode' in features
    print 'Loading BED data'
    bigwig_names, meta_names, datagen_bed, nonblacklist_bools = utils.load_beddata(genome, bed_file, use_meta, use_gencode, input_dir, chrom)
    assert bigwig_names == model_bigwig_names
    if use_meta:
        model_meta_file = model_dir + '/meta.txt'
        assert os.path.isfile(model_meta_file)
        model_meta_names = np.loadtxt(model_meta_file, dtype=str)
        if len(model_meta_names.shape) == 0:
            model_meta_names = [str(model_meta_names)]
        else:
            model_meta_names = list(model_meta_names)
        assert meta_names == model_meta_names
    output_results(bigwig_names, datagen_bed, model, output_dir)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    main()
