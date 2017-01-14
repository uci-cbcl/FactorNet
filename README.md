README for FactorNet
====================

Citing FactorNet
================
Quang, D. and Xie, X. ``FactorNet: a deep learning framework for predicting cell-type specific transcription factor binding from nucleotide-resolution sequential data'', In preparation, 2017.

INSTALL
=======
FactorNet uses several bleeding edge packages that are sometimes not backwards compatible with older code or packages. Therefore, I have included the most recent version numbers of the packages for the configuration that worked for me. For the record, I am using Ubuntu Linux 14.04 LTS with an NVIDIA Titan X GPU (both Maxwell and Pascal architectures).

Required
--------
* [Python] (https://www.python.org) (2.7.10). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.continuum.io) (2.3.0). I listed the versions of Python and Anaconda I used, but the latest versions should be fine. If you're curious as to what packages in Anaconda are used, they are: [numpy] (http://www.numpy.org/) (1.10.1), [scipy] (http://www.scipy.org/) (0.16.0), and [h5py] (http://www.h5py.org) (2.5.0). 
* [Theano] (https://github.com/Theano/Theano) (latest). At the time I wrote this, Theano 0.7.0 is already included in Anaconda. However, it is missing some crucial helper functions. You need to git clone the latest bleeding edge version since there is not a version number for it:

```
$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py develop
```

* [keras] (https://github.com/fchollet/keras/releases/tag/0.2.0) (0.2.0). Deep learning package that uses Theano backend. I'm in the process of upgrading to version 0.3.0 with the Tensorflow backend.

USAGE
=====
