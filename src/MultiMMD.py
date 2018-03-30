import sys
import numpy as np
from keras import backend as K
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
IntType = 'int32'
FloatType = 'float32'

class MultiMMD:
    sample_ratio = 0.75
    tissue_map = {'breast': 0, 'thyroid':1, 'prostate':2}
    tissue_mapper = lambda t: tissue_map[t]
    
    def __init__(self, output_layer, target, target_validation_split=0.1, sample_ratio=0.75,
                 tissue_counts=None scales=None, weights=None, n_neighbors=10):
        print('initializing')
        
        