import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from common import *


def PCA_DWT(X):

    pca = PCA(n_components=PCA_components)

    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # k = np.argmax(cumulative_variance>0.99)
    # print(k)
    reduced_r = pca.transform(X)
    return pca, reduced_r

def inverse_PCA_DWT(pca, reduced_r):

    _, k = reduced_r.shape

    # pca = PCA(n_components=k)
    return pca.inverse_transform(reduced_r)
