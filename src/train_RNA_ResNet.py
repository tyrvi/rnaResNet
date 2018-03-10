import os.path
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from keras.layers import Input, Dense, merge, Activation, add
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
from keras.layers.normalization import BatchNormalization
#detect display
import os
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
import ScatterHist as sh
from keras import initializers
from numpy import genfromtxt
import sklearn.preprocessing as prep
import tensorflow as tf
import keras.backend as K

import pandas as pd
from ggplot import *


class ResNet():
    def __init__(self, layer_sizes=[20, 20], l2_penalty=1e-2):
        self.layer_sizes = layer_sizes
        self.l2_penalty = l2_penalty
    
    def load_data(self, source_path, target_path):        
        self.source_df = pd.read_csv(source_path, sep=',', header=0, index_col=0)
        self.target_df = pd.read_csv(target_path, sep=',', header=0, index_col=0)
        
        self.source = self.source_df.loc[:, "PC1":"PC20"].values
        self.target = self.target_df.loc[:, "PC1":"PC20"].values
        
        self.inputDim = self.target.shape[1]
        
    def init_res_net(self):
        # input
        calibInput = Input(shape=(self.inputDim, ))

        # block 1
        block1_bn1 = BatchNormalization()(calibInput)
        block1_a1 = Activation('relu')(block1_bn1)
        block1_w1 = Dense(self.layer_sizes[1], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1)
        block1_bn2 = BatchNormalization()(block1_w1)
        block1_a2 = Activation('relu')(block1_bn2)
        block1_w2 = Dense(self.layer_sizes[0], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2)
        block1_output = add([block1_w2, calibInput])

        # block 2
        block2_bn1 = BatchNormalization()(block1_output)
        block2_a1 = Activation('relu')(block2_bn1)
        block2_w1 = Dense(self.layer_sizes[1], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1)
        block2_bn2 = BatchNormalization()(block1_w1)
        block2_a2 = Activation('relu')(block2_bn2)
        block2_w2 = Dense(self.layer_sizes[0], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2)
        block2_output = add([block2_w2, calibInput])

        # block 3
        block3_bn1 = BatchNormalization()(block2_output)
        block3_a1 = Activation('relu')(block3_bn1)
        block3_w1 = Dense(self.layer_sizes[1], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1)
        block3_bn2 = BatchNormalization()(block3_w1)
        block3_a2 = Activation('relu')(block3_bn2)
        block3_w2 = Dense(self.layer_sizes[0], activation='linear', kernel_regularizer=l2(self.l2_penalty), 
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2)
        block3_output = add([block3_w2, calibInput])

        self.calibMMDNet = Model(inputs=calibInput, outputs=block3_output)
        
        def step_decay(epoch):
            initial_lrate = 0.1
            drop = 0.1
            epochs_drop = 250.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        
        self.lrate = LearningRateScheduler(step_decay)
        optimizer = keras.optimizers.Adam()

        self.calibMMDNet.compile(optimizer=optimizer, 
                            loss=lambda y_true,y_pred: 
                               cf.MMD(block3_output, self.target, MMDTargetValidation_split=0.1,
                                     MMDTargetSampleSize=100, n_neighbors=10).KerasCost(y_true, y_pred)
                           )

        K.get_session().run(tf.global_variables_initializer())

        self.source_labels = np.zeros(self.source.shape)
        print(self.source_labels.shape)
        
        
    def train(self, epochs=2000, batch_size=20, validation_split=0.1, verbose=1,
             callbacks=[]):
        
        # self.lrate, cb.EarlyStopping(monitor='val_loss', patience=50, mode='auto')
        
        self.calibMMDNet.fit(self.source, self.source_labels, epochs=epochs, batch_size=batch_size,
                             validation_split=validation_split, verbose=verbose, callbacks=callbacks)
        
        self.calibrated_source = self.calibMMDNet.predict(self.source)
        self.calibrated_source_df = pd.DataFrame(self.calibrated_source, 
                                                 index=self.source_df.index, columns=self.source_df.columns[2:])
        self.calibrated_source_df.reindex(columns=self.source_df.columns)
        self.calibrated_source_df['study'] = self.source_df['study']
        self.calibrated_source_df['tissue'] = self.source_df['tissue']
        
    def pca(self):
        pca = decomposition.PCA()
        
        # data = np.append(self.target, self.source, axis=0)
        
        pca.fit(self.target)
        #pca.fit(data)
        
        self.target_sample_pca = pca.transform(self.target)
        self.projection_before = pca.transform(self.source)
        self.projection_after = pca.transform(self.calibrated_source)
        
        self.target_pca_df = pd.DataFrame(self.target_sample_pca,
                                          index=self.target_df.index, columns=self.target_df.columns[2:])
        self.target_pca_df.reindex(columns=self.target_df.columns)
        self.target_pca_df['study'] = self.target_df['study']
        self.target_pca_df['tissue'] = self.target_df['tissue']
        
        self.source_pca_df = pd.DataFrame(self.projection_before,
                                          index=self.source_df.index, columns=self.source_df.columns[2:])
        self.source_pca_df.reindex(columns=self.source_df.columns)
        self.source_pca_df['study'] = self.source_df['study']
        self.source_pca_df['tissue'] = self.source_df['tissue']
        
        self.calibrated_source_pca_df = pd.DataFrame(self.projection_after,
                                          index=self.source_df.index, columns=self.source_df.columns[2:])
        self.calibrated_source_pca_df.reindex(columns=self.source_df.columns)
        self.calibrated_source_pca_df['study'] = self.source_df['study']
        self.calibrated_source_pca_df['tissue'] = self.source_df['tissue']
        
        
    def scatter_hist(self, pc1=0, pc2=1):
        axis1 = 'PC' + str(pc1)
        axis2 = 'PC' + str(pc2)
        sh.scatterHist(self.target_sample_pca[:, pc1], self.target_sample_pca[:, pc2], self.projection_before[:, pc1], 
                       self.projection_before[:, pc2], axis1, axis2)
        sh.scatterHist(self.target_sample_pca[:, pc1], self.target_sample_pca[:, pc2], self.projection_after[:, pc1], 
                       self.projection_after[:, pc2], axis1, axis2)
        
    def scatter_plot(self):        
        df1 = pd.concat([self.target_pca_df, self.source_pca_df])
        p1 = ggplot(df1, aes(x='PC1', y="PC2", color='tissue', shape='study')) + geom_point()
        print(p1)
        
        df2 = pd.concat([self.target_pca_df, self.calibrated_source_pca_df])
        p2 = ggplot(df2, aes(x='PC1', y="PC2", color='tissue', shape='study')) + geom_point()
        print(p2)
        
        
    def save_calibrated(self, calibrated_path=''):
        # df = pd.DataFrame(self.calibrated_source)
        
        self.calibrated_source_df.to_csv(calibrated_path)
        
                        