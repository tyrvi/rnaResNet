'''
Created on Dec 5, 2016

@author: urishaham
'''

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


# configuration hyper parameters
denoise = False # whether or not to train a denoising autoencoder to remove the zeros
keepProb=.8

# AE confiduration
ae_encodingDim = 25
l2_penalty_ae = 1e-2 

#MMD net configuration
mmdNetLayerSizes = [25, 25]
l2_penalty = 1e-2
#init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)
#def my_init (shape):
#    return initializers.normal(stddev=.1e-4)
#my_init = 'glorot_normal'

#######################
###### read data ######
#######################
# we load two CyTOF samples 

data = 'person1_3month'

if data =='person1_baseline':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_baseline.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_baseline.csv')
if data =='person2_baseline':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_baseline.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_baseline.csv')
if data =='person1_3month':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_3month.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_3month.csv')
if data =='person2_3month':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_3month.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_3month.csv')
   
source = genfromtxt(sourcePath, delimiter=',', skip_header=0)
target = genfromtxt(targetPath, delimiter=',', skip_header=0)

# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source) 

numZerosOK=1
toKeepS = np.sum((source==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS))
toKeepT = np.sum((target==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT))

inputDim = target.shape[1]

if denoise:
    trainTarget_ae = np.concatenate([source[toKeepS], target[toKeepT]], axis=0)
    np.random.shuffle(trainTarget_ae)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    encoded1 = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(encoded)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded1)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, epochs=500, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])    
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
target = preprocessor.transform(target)    

#############################
######## train MMD net ######
#############################


calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2) 
block1_output = add([block1_w2, calibInput])
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2) 
block2_output = add([block2_w2, block1_output])
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1) 
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(inputDim, activation='linear',kernel_regularizer=l2(l2_penalty), 
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2) 
block3_output = add([block3_w2, block2_output])

calibMMDNet = Model(inputs=calibInput, outputs=block3_output)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

#train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block3_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
K.get_session().run(tf.global_variables_initializer())

sourceLabels = np.zeros(source.shape[0])
calibMMDNet.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate, mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

##############################
###### evaluate results ######
##############################

calibratedSource = calibMMDNet.predict(source)

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)
 
'''
# save models
autoencoder.save(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_DAE.h5'))                 
calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  
'''
