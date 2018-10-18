import keras.optimizers
from keras.layers import Input, Dense, merge, Activation, add
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
from keras.layers.normalization import BatchNormalization
import CostFunctions as cf
import MultiMMD as m
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
import ScatterHist as sh
from keras import initializers
import tensorflow as tf
import keras.backend as K
import pandas as pd


class DataSet:
    source_path = None
    target_path = None
    source_df = None
    target_df = None
    source = None
    target = None

    def __init__(self, source_path=None, target_path=None):
        assert (source_path is not None and target_path is not None), "Must provide source and target paths"

        self.source_path = source_path
        self.target_path = target_path

        self.source_df = pd.read_csv(source_path, sep=',', header=0, index_col=0)
        self.target_df = pd.read_csv(target_path, sep=',', header=0, index_col=0)

        self.source = self.source_df.loc[:, "PC1":].values
        self.target = self.target_df.loc[:, "PC1":].values


def create_model(layer_sizes=[20, 20], l2_penalty=1e-2, input_dim=1,
                 optimizer=None, loss=None):
    assert (optimizer is not None), "must provide an optimizer"
    assert (loss is not None), "must provide a loss"

    # input
    calibInput = Input(shape=(inputDim, ))

    # block 1
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(layer_sizes[1], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1)
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(layer_sizes[0], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2)
    block1_output = add([block1_w2, calibInput])
    
    # block 2
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(layer_sizes[1], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1)
    block2_bn2 = BatchNormalization()(block1_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(layer_sizes[0], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2)
    block2_output = add([block2_w2, block1_output])
    
    # block 3
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(layer_sizes[1], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1)
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(layer_sizes[0], activation='linear', kernel_regularizer=l2(l2_penalty), kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2)
    block3_output = add([block3_w2, block2_output])
    
    calibMMDNet = Model(inputs=calibInput, outputs=block3_output)
    calibMMDNet.compile(optimizer=optimizer, loss=loss)

    return calibMMDNet
