# README for FactorNet

README still under construction (1/30/17). As you can tell, there are a lot of similarly named files. Many lines of code were embarassingly copied in multiple scripts instead of simply shared. I will try to condense everything into 2 or 3 script files soon, but everything I used for the ENCODE-DREAM competition is here.

# Citing FactorNet

Quang, D. and Xie, X. ``FactorNet: a deep learning framework for predicting cell-type specific transcription factor binding from nucleotide-resolution sequential data'', In preparation, 2017.

# INSTALL

FactorNet uses several bleeding edge packages that are sometimes not backwards compatible with older code or packages. Therefore, I have included the most recent version numbers of the packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 14.04 LTS with an NVIDIA Titan X GPU (both Maxwell and Pascal architectures) and 128 GBs of RAM. Theoretically, about 24 GB of ram is needed to train one model, since the current version loads two copies of the genome into memory for efficient sequence extraction. If you would like a slightly slower version that saves memory by reading the genome from hard disk instead of RAM, let me know.

## Required
* [Python] (https://www.python.org) (2.7.11). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.continuum.io) (4.0.0). I listed the versions of Python and Anaconda I used, but the latest versions should be fine. If you're curious as to what packages in Anaconda are used, they are: [numpy] (http://www.numpy.org/) (1.10.4), [scipy] (http://www.scipy.org/) (0.17.0). Standard python packages are: sys, os, errno, argparse, pickle and itertools. 
* [Theano] (https://github.com/Theano/Theano) (latest). At the time I wrote this, versoin 0.8.2 was the latest release of Theano. However, it is incompatible with the latest versions of CUDA (8.0) and cuDNN (5.1) I was using. You need to git clone the latest bleeding edge version since there is not a version number for it:
```
$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py develop
```

* [pyfasta] (https://pypi.python.org/pypi/pyfasta) (0.5.2). Loads the hg19.fa FASTA file in the resources folder.

* [pyBigWig] (https://github.com/dpryan79/pyBigWig/releases/tag/0.2.8) (0.2.8). Efficiently loads bigWig continuous values into memory. Newer versions should be fine.

* [pybedtools] (https://pypi.python.org/pypi/pybedtools) (0.7.8). Also requires [bedtools2] (https://github.com/arq5x/bedtools2).

* [parmap] (https://pypi.python.org/pypi/parmap/1.3.0) (1.3.0). Helps parallelize certain functions like preprocessing multiple BED files.


* [keras] (https://github.com/fchollet/keras/releases/tag/1.1.1) (1.1.1). Deep learning package that uses Theano backend. Newer versions are likely to be fine, but Keras has a history of not being backwards compatible with older versions.

## Optional

* [CUDA] (https://developer.nvidia.com/cuda-toolkit) (8.0). Theano can use either CPU or GPU, but using a GPU is almost entirely necessary for a network and dataset this large.

* [cuDNN] (https://developer.nvidia.com/cudnn) (5.1). Significantly speeds up convolution and recurrent operations. 

# USAGE

## Data

Before training, you must have a copy of the hg19 genome FASTA, hg19.fa, in the resources folder. Due to space issues, the FASTA file is not included in this repository. This file also cannot be compressed, so you will have to deal with it taking up quite a bit of space on the hard drive. You can get yourself a copy of the file with the following command lines:

```
$ wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz
$ tar zxvf chromFa.tar.gz
$ cat chr*.fa > hg19.fa
```

All training data are in the data folder. Training data are organized into one folder per cell line. The README in the data folder will give you an idea of how to format cell data if you want to use FactorNet on your own data. You will also need download additional files into the data folder before you can proceed with the example code below.

## Training (train.py)

### Arguments

The following are arguments for train.py, the training script:
* `-i inputdirs`. Folder(s) containing cell-type specific data for training. Examples and a more in-depth explanation can be found in the data folder (required).
* `-vi validinputdir`. A folder containing cell-type specific data for validation (Optional. If not specified, validation chromosomes in the training cell lines will be used instead).
* `-v validchroms`. Chromosome(s) to set aside for validation or early stopping (default: chr11).
* `-t testchroms`. Chromosome(s) to set aside for testing. Test sequences are never touched throughout training (default: chr1, chr8, chr21).
* `-e epochs`. Number of epochs for training (default: 20)
* `-n negatives`. Number of negative samples per each positive sample (default: 1). For example, if this value is set to 5 and you are training on a ChIP-seq file with 10,000 peaks, then each epoch will contain 10,000 positive samples and 50,000 randomly drawn negative samples without replacement. Not used for multi-task training.
* `-k kernels`. Number of kernels or motifs in the model (default: 32).
* `-r recurrent`. Number of LSTM cells in the model (default: 32).
* `-d dense`. Number of dense units in the penultimate layer (default: 64).
* `-p dropout`. Dropout rate between the LSTM and dense layers (defaulter: 0.5).
* `-s seed`. Random seed for reproducibility (default: 420).
* `-f factor`. The transcription factor to train. If not specified, multi-task training is used instead.
* `-m meta`. Meta flag. If used, model will use metadata features.
* `-g gencode`. GENCODE flag. If used, model will incorporate CpG island and gene annotation features.
* `-o outputdir`. Output directory. Will not be overwritten if already present (required, unless `-oc` used).
* `-oc outputdirc`. Output directory. Will be overwritten if already present (required, unless `-o` used).

### Examples
There are five different type of models used for the ENCODE-DREAM competition. These models differ in their training method and types of features they incorporate. I included a sample BED file in the resources folder (sample_ladder_regions.blacklistfiltered.bed.gz) to make predictions on. The README in the models folder describe what the program outputs once training completes.

The following are descriptions and code snppets for each model.

* multiTask
This model trains with multiple TF binding targets in a joint multi-task fashion. Each epoch, it randomly draws a number of negative bins (with replacement) equal to the number of positive bins. Bins labeled as ambiguous are treated as negative bins. This model can only train on one cell line.
```
$ python train.py -i data/HepG2 -k 128 -r 128 -d 256 -oc multiTask_Unique35_DGF_HepG2
$ # If you want to remove the 35 bp Uniqueness track as features, remove that line from the bigwig.txt file in data/HepG2.
$ # Of all the ENCODE-DREAM training cell lines, HepG2 has the most TFs associated with it (19), necessitating a bigger model.
```

* onePeak
The onePeak model, and all subsequent models, are trained on a single TF in a single-task fashion. You can initiate single-task training with the `-f` command. Unlike the multiTask model, single-task models can leverage data from multiple reference cell lines and ignores ambiguous peaks. The next three models are more or less variations of this model, but modified to incorporate non-sequential metadata features such as gene expression and gene annotations.
```
$ python train.py -f MAX -k 128 -r 64 -d 128 -n 3 -e 5 -i data/A549 data/GM12878 data/H1-hESC data/HCT116 data/HeLa-S3 data/HepG2 data/K562 -oc onePeak_Unique35_DGF_3n_50e_128k_64r_128d_MAX
```

* meta
This model incorporates cell-type specific features such as gene expression. These features are specified in the meta.txt file in each data folder. Use the `-m` option if you want to use this model
```
$ python train.py -f GABPA -n 5 -m -i data/GM12878 data/H1-hESC data/HeLa-S3 data/HepG2 data/MCF-7 -oc meta_Unique35_DGF_5n_GABPA 
```

* GENCODE
This model is very similar to the meta model. Like the meta model, it incorporates non-sequential metadata features, except at the bin level instead of at the cell type level. Bins are annotated with GENCODE (promoters, introns, 5' UTR, 3' UTR, and CDS) and unmasked CpG islands. Promoters are defined as genomic regions up to 300 bps upstream and 100 bps downstream of a TSS. Use the `-g` option if you want to use this model.

```
$ python train.py -f REST -k 64 -d 128 -g -i data/H1-hESC data/HeLa-S3 data/HepG2 data/MCF-7 data/Panc1 -oc GENCODE_Unique35_DGF_64k_128d_REST
```

* metaGENCODE
This model includes the metadata features from both the meta and GENCODE models. Combing the `-g` and `-m` options to use this model
```
$ python train.py -f CTCF -k 128 -r 64 -d 128 -m -g -i data/A549 data/H1-hESC data/HeLa-S3 data/HepG2 data/IMR-90 data/K562 data/MCF-7 -oc metaGENCODE_Unique35_DGF_129k_64r_128d_CTCF
```

## Predicting (predict.py)

### Arguments

The following are arguments for predict.py, the prediction script:
* `-i inputdir`. A folder containing cell-type specific data to perform predictions on (required).
* `-m modeldir`. A older containing trained model generated by train.py (required).
* `-f factor`. The transcription factor to evaluate (required).
* `-b bed`. BED file containing intervals to predict on (required).
* `-o outputfile`. The output filename of the gzipped bedgraph file (required).

### Examples
Using the prediction script is fairly straightforward. From the model folder, it will recognize what kind of model it is. I included a sample BED file in the resources folder (sample_ladder_regions.blacklistfiltered.bed.gz) to make predictions on. The following command line is a pretty typical use of the script:

```
$ python predict.py -f CTCF -i data/PC-3 -m models/CTCF/metaGENCODE_RNAseq_Unique35_DGF -b resources/sample_ladder_regions.blacklistfiltered.bed.gz -o CTCF_PC-3.bed.gz
```
## Visualizing (visualize.py)

### Arguments

The following are arguments for visualize.py, the visualization script:
* `-i inputdir`. A folder containing cell-type specific data to perform predictions on (required).
* `-m modeldir`. A older containing trained model generated by train.py (required).
* `-f factor`. The transcription factor to evaluate (required).
* `-b bed`. BED file containing intervals to predict on (required).
* `-c chrom`. Chromosome to use for visualization. Only sequences on this chromosome will be used (default: chr11).
* `-o outputfile`. The output filename of the gzipped bedgraph file (required).

### Examples

The visualization script applies heuristics to convert the kernels into sequence motifs. It also helps with plotting the mean bigWig signals of the kernels by outputting these values to a numpy .npy file. It will work on the first layer (a convolutional layer) and the second layer (a time distributed dense layer). 

```
$ python visualize.py -i data/HepG2 -b data/HepG2/ChIPseq.HepG2.REST.conservative.train.narrowPeak.gz -m models/REST/GENCODE_Unique35_DGF_2 -oc GENCODE_Unique35_DGF_2_visualizations
```
