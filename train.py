#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, add_shared_layer, Layer, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.layer_utils import model_summary
import pickle
