from rna_resnet import ResNet
from keras import callbacks as cb
from Calibration_Util import FileIO as io
import os
import numpy as np
import CostFunctions as cf
from keras import backend as K
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from plots import heatmap
from sys import argv


def train_net(epochs=1000):
    folder = '../data/6pc/'
    # GTEX as source and TCGA as target
    source_file = 'unnorm-log-6PC-GTEX-breast-prostate-thyroid.csv'
    target_file = 'unnorm-log-6PC-TCGA-breast-prostate-thyroid.csv'

    source_path = folder + source_file
    target_path = folder + target_file

    rnaNet = ResNet(layer_sizes=[6, 20])

    rnaNet.load_data(source_path=source_path, target_path=target_path)

    print("gtex = source shape = " + str(rnaNet.source.shape))
    print("tcga = target shape = " + str(rnaNet.target.shape))

    rnaNet.init_res_net()

    callbacks = [rnaNet.lrate, cb.EarlyStopping(monitor='val_loss', patience=100, mode='auto')]

    rnaNet.train(epochs=epochs, callbacks=callbacks, batch_size=50)
    print("finished training")

    print("Running sanity check...")
    rnaNet.predict()
    source = rnaNet.source.astype('float32')
    target = rnaNet.target.astype('float32')
    calibrated_source = rnaNet.calibrated_source.astype('float32')

    mmd = cf.MMD(source, target, MMDTargetSampleSize=target.shape[0], n_neighbors=10)
    mmd_before = K.eval(mmd.cost(source, target))
    mmd_after = K.eval(mmd.cost(calibrated_source, target))

    # print("MMD before: %0.10f" % mmd_before)
    # print("MMD after: %0.10f" % mmd_after)

    if mmd_after < mmd_before:
        print("PASSED sanity check\n")
        return rnaNet, 0, mmd_before, mmd_after
    else:
        print("FAILED sanity check\n")
        return rnaNet, 1, mmd_before, mmd_after


def main(params):
    models = []
    model = {
        'rnaNet': None,
        'error': 0,
        'mmd_before': -1,
        'mmd_after': -1
    }

    print("Training {} models...\n".format(params['N']))
    num_failed = 0
    rnaNet, error, mmd_before, mmd_after = train_net()
    model['rnaNet'] = rnaNet
    model['error'] = error
    model['mmd_before'] = mmd_before
    model['mmd_after'] = mmd_after
    num_failed += error

    print("FAILED training {} models\n".format(num_failed))

    data = model['rnaNet'].source
    print('model {} MMD before = {}; MMD after = {}'.format(i, model['mmd_before'], model['mmd_after']))
    for i in range(params['N']):
        model['rnaNet'].predict(data=data)
        data = model['rnaNet'].calibrated_source

    source = model['rnaNet'].source.astype('float32')
    target = model['rnaNet'].target.astype('float32')
    calibrated_source = model['rnaNet'].calibrated_source.astype('float32')

    mmd = cf.MMD(source, target, MMDTargetSampleSize=target.shape[0], n_neighbors=10)
    mmd_before = K.eval(mmd.cost(source, target))
    mmd_after = K.eval(mmd.cost(calibrated_source, target))

    print("\nMMD before: %0.10f" % mmd_before)
    print("MMD after: %0.10f" % mmd_after)

    return


if __name__ == "__main__":
    params = {
        'N': 3 if len(argv) < 2 else int(argv[1])
    }

    main(params)
