import sys
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
from math import floor
IntType = 'int32'
FloatType = 'float32'


def calculate_ranges(df_counts):
    num_tissues = df_counts.shape[0]
    ranges = np.zeros((num_tissues, 2), dtype=IntType)

    low = 0
    for i in range(num_tissues):
        tissue = df_counts.index[i]
        high = low + df_counts[i]
        print("tissue = {0}, low = {1}, high = {2}".format(tissue, low, high))
        ranges[i] = [low, high]
        low = high

    return ranges


def squared_distance(X, Y):
    # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
    # broadcasted subtraction, a square, and a sum.
    r = K.expand_dims(X, axis=1)
    return K.sum(K.square(r-Y), axis=-1)


class MultiMMD:
    output_layer = None
    target_train = None
    target_validate = None
    sample_ratio = 0.75
    tissue_map = {'breast': 0, 'thyroid': 1, 'prostate': 2}
    tissue_mapper = lambda t: tissue_map[t]
    kernel = None
    scales = None
    weights = None
    n_neighbors = None

    def __init__(self,
                 output_layer,
                 target_df,
                 target_validation_split=0.1,
                 target_sample_size=100,
                 sample_ratio=0.75,
                 scales=None,
                 weights=None,
                 n_neighbors=10):

        target_df = target_df.sort_values(['tissue'])
        target_df_labels = target_df.loc[:, 'tissue']
        target_df_counts = target_df['tissue'].value_counts().sort_index()
        target = target_df.loc[:, "PC1":].values

        if scales == None:
            print("setting scales using KNN")
            num_tissues = target_df_counts.shape[0]
            scales = np.zeros((num_tissues, 3))

            # calculate tissue ranges in target
            ranges = calculate_ranges(target_df_counts)
            print("")

            for t in range(num_tissues):
                tissue = target_df_counts.index[t]
                print("setting scales for tissue {0} {1}".format(t, tissue))
                med = np.zeros(20)
                sample_size = floor(sample_ratio * target_df_counts[t])
                for ii in range(0, 20):
                    low = ranges[t, 0]
                    high = ranges[t, 1]
                    ix = np.random.randint(low=low, high=high, size=sample_size)
                    sample = target[ix, :]
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
                    distances, dummy = nbrs.kneighbors(sample)
                    med[ii] = np.median(distances[:, 1:n_neighbors])
                med = np.median(med)
                scales[t] = [med/2, med, 2*med]
                print("{0}\n".format(scales[t]))

        self.scales = K.variable(value=scales)

        if weights == None:
            print("setting all scale weights to 1")
            weights = K.eval(K.shape(scales)[0])

        self.weights = K.variable(value=np.asarray(weights))

        # split target into train and validation sets
        target_train_df, target_validate_df = train_test_split(target_df, test_size=0.1,
                                                               random_state=42)
        # sort and extract labels
        target_train_df = target_train_df.sort_values(['tissue'])
        target_validate_df = target_validate_df.sort_values(['tissue'])
        self.target_train_labels = target_train_df.loc[:, 'tissue']
        self.target_validate_labels = target_validate_df.loc[:, 'tissue']

        # extract target and validate tissue counts for determining ranges
        self.target_train_counts = target_train_df['tissue'].value_counts().sort_index()
        self.target_validate_counts = target_validate_df['tissue'].value_counts().sort_index()

        print("\ntarget train counts")
        print(self.target_train_counts)
        print("\ntarget validation counts")
        print(self.target_validate_counts)

        self.target_train = target_train_df.loc[:, "PC1":].values.astype(FloatType)
        # self.target_train = self.target_train.astype(FloatType)
        self.target_validate = target_validate_df.loc[:, "PC1":].values.astype(FloatType)
        # self.target_validate = self.target.validate.astype(FloatType)

        print("\ntarget train shape")
        print(self.target_train.shape)
        print("\ntarget validate shape")
        print(self.target_validate.shape)
        print("")

        print("calculating training ranges")
        self.target_train_ranges = calculate_ranges(self.target_train_counts)
        print("\ncalculating validation ranges")
        self.target_validate_ranges = calculate_ranges(self.target_validate_counts)

        self.target_sample_size = target_sample_size
        self.sample_ratio = sample_ratio
        self.n_neighbors = n_neighbors
        self.num_tissues = self.target_train_counts.shape[0]
        self.kernel = self.RaphyKernel
        self.output_layer = output_layer

    def RaphyKernel(self, X, Y, tissue_num):
        # returns a 1xnxm tensor where 1 is broadcastable
        sq_dist = K.expand_dims(squared_distance(X, Y), 0)
        # extract the scales for the given tissue
        scales = K.gather(self.scales, tissue_num)
        # expand scales into a px1x1 tensor for element wise exponential
        scales = K.expand_dims(K.expand_dims(scales, -1), -1)
        # expand weights into a px1x1 tensor for element wise exponential
        weights = K.expand_dims(K.expand_dims(self.weights, -1), -1)

        # calculate the kernel for each scale weight on the distance matrix and sum
        return K.sum(weights * K.exp(-sq_dist / (K.pow(scales, 2))), 0)

    def cost(self, source, target, tissue_num):
        # calculate 3 MMD terms
        xx = self.kernel(source, source, tissue_num)
        xy = self.kernel(source, target, tissue_num)
        yy = self.kernel(target, target, tissue_num)
        # calculate the MMD estimate
        MMD = K.mean(xx) - 2*K.mean(xy) + K.mean(yy)
        # if we get NaN then there was no source so we make it 0
        MMD = np.nan_to_num(K.eval(MMD))
        MMD = K.cast(MMD, FloatType)
        # return sqrt for efficiency
        return K.sqrt(MMD)

    def KerasCost(self, y_true, y_pred):
        ret = 0
        for t in range(self.num_tissues):
            # sample from target tissues
            low = self.target_train_ranges[t, 0]
            high = self.target_train_ranges[t, 1]
            sample_size = floor(self.sample_ratio * self.target_train_counts[t])
            sample_train = K.cast(K.round(K.random_uniform_variable(shape=tuple([sample_size]),
                                                                    low=low, high=high)), IntType)
            sample_target_train = K.gather(self.target_train, sample_train)

            # select validation samples (use all)
            low = self.target_validate_ranges[t, 0]
            high = self.target_validate_ranges[t, 1]
            sample_validate = K.cast(np.fromiter((i for i in range(low, high)), dtype=IntType),
                                     IntType)
            sample_target_validate = K.gather(self.target_validate, sample_validate)

            source_labels = K.eval(y_true)
            source_index = np.where(np.isin(source_labels, t))[0]
            source = K.eval(self.output_layer)[source_index]
            # source = K.cast_to_floatx(source)

            target = K.in_train_phase(sample_target_train, sample_target_validate)
            # target = sample_target_train

            ret += self.cost(source, target, t)

        ret = ret + 0*K.cast(K.sum(y_pred), FloatType) + 0*K.cast(K.sum(y_true), FloatType)
        return ret
