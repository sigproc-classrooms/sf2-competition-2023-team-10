import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from common import *


def PCA_DWT(X):

    pca = PCA(n_components=PCA_components)

    reduced_r = pca.fit_transform(X)

    return pca, reduced_r

def inverse_PCA_DWT(pca, reduced_r):

    _, k = reduced_r.shape

    # pca = PCA(n_components=k)
    return pca.inverse_transform(reduced_r)
