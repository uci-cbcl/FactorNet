README for FactorNet
====================

README still under construction (1/26/17). As you can tell, there are a lot of similarly named files. Many lines of code were embarassingly copied in multiple scripts instead of simply shared. I will try to condense everything into 2 or 3 script files soon, but everything I used for the ENCODE-DREAM competition is here.

Citing FactorNet
================
Quang, D. and Xie, X. ``FactorNet: a deep learning framework for predicting cell-type specific transcription factor binding from nucleotide-resolution sequential data'', In preparation, 2017.

INSTALL
=======
FactorNet uses several bleeding edge packages that are sometimes not backwards compatible with older code or packages. Therefore, I have included the most recent version numbers of the packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 14.04 LTS with an NVIDIA Titan X GPU (both Maxwell and Pascal architectures) and 128 GBs of RAM. Theoretically, about 24 GB of ram is needed to train one model, since the current version loads two copies of the genome into memory for efficient sequence extraction. If you would like a slightly slower version that saves memory by reading the genome from hard disk instead of RAM, let me know.

Required
--------
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

Optional
--------
* [CUDA] (https://developer.nvidia.com/cuda-toolkit) (8.0). Theano can use either CPU or GPU, but using a GPU is almost entirely necessary for a network and dataset this large.

* [cuDNN] (https://developer.nvidia.com/cudnn) (5.1). Significantly speeds up convolution and recurrent operations. 

USAGE
=====
Data
----

The README in the data folder will give you an idea of how to format cell data if you want to use FactorNet on your own data.

Training
--------

The README in the models folder will give you an idea of what the program outputs once training completes.

Prediction
----------

```
$ python predict.py -m -b -o
```


