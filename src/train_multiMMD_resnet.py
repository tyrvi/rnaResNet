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


def main():
    rnaNet = ResNet(layer_sizes=[6, 20])

    folder = '../data/6pc/'
    # GTEX as source and TCGA as target
    source_file = 'unnorm-log-6PC-GTEX-breast-prostate-thyroid.csv'
    target_file = 'unnorm-log-6PC-TCGA-breast-prostate-thyroid.csv'

    source_path = folder + source_file
    target_path = folder + target_file

    rnaNet.load_data(source_path=source_path, target_path=target_path)

    print("gtex = source shape = " + str(rnaNet.source.shape))
    print("tcga = target shape = " + str(rnaNet.target.shape))

    rnaNet.init_res_net()

    callbacks=[rnaNet.lrate, cb.EarlyStopping(monitor='val_loss', patience=100, mode='auto')]
    rnaNet.train(epochs=1000, callbacks=callbacks, batch_size=50)
    print("finished training")
    rnaNet.predict()
    rnaNet.pca()
    print("finished pca")
    
    # plt.style.use('ggplot')

    # df = pd.concat([rnaNet.source_pca_df, rnaNet.target_pca_df])
    # ax = sns.scatterplot(x='PC1', y='PC2', data=df, hue='tissue', style='study')
    # ax.set_title('before')
    # ax.figure.savefig('../plots/pca_before.pdf')
    # plt.close()

    # df = pd.concat([rnaNet.calibrated_source_pca_df, rnaNet.target_pca_df])
    # ax = sns.scatterplot(x='PC1', y='PC2', data=df, hue='tissue', style='study')
    # ax.set_title('after')
    # ax.figure.savefig('../plots/pca_after.pdf')
    # plt.close()

    # df = pd.concat([rnaNet.source_df, rnaNet.target_df])
    # df = df.drop(["study", "tissue"], axis=1).transpose().corr()
    # ax = sns.heatmap(df, xticklabels=False, yticklabels=False)
    # ax.set_title("before")
    # ax.figure.savefig('../plots/heatmap_before.png')
    # plt.close()

    # df = pd.concat([rnaNet.calibrated_source_df, rnaNet.target_df])
    # df = df.drop(["study", "tissue"], axis=1).transpose().corr()
    # ax = sns.heatmap(df, xticklabels=False, yticklabels=False)
    # ax.set_title("after")
    # ax.figure.savefig('../plots/heatmap_after.png')
    # plt.close()

    source = rnaNet.source.astype('float32')
    target = rnaNet.target.astype('float32')
    calibrated_source = rnaNet.calibrated_source.astype('float32')

    mmd = cf.MMD(source, target, MMDTargetSampleSize=target.shape[0], n_neighbors=10)
    mmd_before = K.eval(mmd.cost(source, target))
    mmd_after = K.eval(mmd.cost(calibrated_source, target))

    print("MMD before: %0.10f" % mmd_before)
    print("MMD after: %0.10f" % mmd_after)

    # folder = 'data/unnorm/'
    # save_file = 'calibrated-unnorm-log-6PC-GTEX-breast-prostate-thyroid.csv'
    # save_path = os.path.join(io.DeepLearningRoot(), folder + save_file)

    # rnaNet.save_calibrated(path='../plots/calibrated_unnorm_log_6pc.csv')


if __name__ == "__main__":
    main()
