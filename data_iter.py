import numpy as np
import pyBigWig
from keras.preprocessing.image import Iterator


# Modified from keras
class DataIterator(Iterator):
    def __init__(self, data_list, genome, batch_size, seqlen, bigwig_rc_order=None, shuffle=False, seed=1337):
        self.data_list = data_list
        if data_list is None or len(data_list) == 0:
            self.num_bigwigs = 0
        else:
            self.num_bigwigs = len(data_list[0][4])
            self.num_meta = len(data_list[0][5])
        if bigwig_rc_order is None:
            self.bigwig_rc_order = np.arange(self.num_bigwigs)
        else:
            self.bigwig_rc_order = bigwig_rc_order
        self.genome = genome
        self.seqlen = seqlen
        self.nucleotides = np.array(['A', 'C', 'G', 'T'])
        if data_list is None or len(data_list) == 0:
            self.labeled = False
        else:
            self.labeled = len(data_list[0]) == 7
        if self.labeled:
            self.num_tfs = len(data_list[0][6])
        super(DataIterator, self).__init__(len(data_list), batch_size, shuffle, seed)

    def __len__(self):
        return len(self.data_list)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_X_seq = np.zeros((current_batch_size, self.seqlen, 4), dtype=bool)
        batch_X_bigwig = np.zeros((current_batch_size, self.seqlen, self.num_bigwigs), dtype=np.float32)
        if self.num_meta:
            batch_X_meta = np.zeros((current_batch_size, self.num_meta), dtype=np.float32)
        if self.labeled:
            batch_y = np.zeros((current_batch_size, self.num_tfs), dtype=bool)
        for i, j in enumerate(index_array):
            data = self.data_list[j]

            chrom = data[0]
            start = data[1]
            stop = data[2]
            shift = data[3]
            bigwig_files = data[4]
            meta = data[5]
            if shift:
                s = np.random.randint(-shift, shift+1)
                start += s
                stop += s
            med = (start + stop) / 2
            start = med - self.seqlen / 2
            stop = med + self.seqlen / 2
            batch_X_seq[i] = self.genome[chrom][start:stop]
            if self.num_meta:
                batch_X_meta[i] = meta
            for k, bigwig_file in enumerate(bigwig_files):
                bigwig = pyBigWig.open(bigwig_file)
                sample_bigwig = np.array(bigwig.values(chrom, start, stop))
                bigwig.close()
                sample_bigwig[np.isnan(sample_bigwig)] = 0
                batch_X_bigwig[i, :, k] = sample_bigwig
            if self.labeled:
                batch_y[i] = data[6]
                # otherwise the binding code is 'U', so leave as 0
        batch_X_seq_rc = batch_X_seq[:, ::-1, ::-1]
        batch_X_bigwig_rc = batch_X_bigwig[:, ::-1, self.bigwig_rc_order]
        batch_X_fwd = np.concatenate([batch_X_seq, batch_X_bigwig], axis=-1)
        batch_X_rev = np.concatenate([batch_X_seq_rc, batch_X_bigwig_rc], axis=-1)
        if self.num_meta:
            batch_x = [batch_X_fwd, batch_X_rev, batch_X_meta]
        else:
            batch_x = [batch_X_fwd, batch_X_rev]
        if self.labeled:
            return batch_x, batch_y
        return batch_x
