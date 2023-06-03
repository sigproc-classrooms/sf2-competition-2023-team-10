import numpy as np
from sklearn.decomposition import PCA
from common import *
from my_LP import quantise


def PCA_DWT_old(X):

    pca = PCA(n_components=PCA_components)

    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    reduced_r = pca.transform(X)
    return reduced_r

def inverse_PCA_DWT_old(reduced_r):
    pca = PCA(n_components=PCA_components)
    return pca.inverse_transform(reduced_r)

def SVD(X):

    u, s, vh = np.linalg.svd(X)

    u = u[:, :svd_dimension]*1023
    s = s[:svd_dimension]
    vh = vh[:svd_dimension, :]*1023

    svd_result = np.concatenate((u, s[np.newaxis, :], vh.T), axis = 0)

    return quantise(svd_result, 1)

def inverse_SVD(coefficients):

    u = coefficients[:256]/1023
    s = coefficients[256]
    vh = coefficients[257:]/1023
    vh = vh.T
    print(u.shape, s.shape, vh.shape)
    return u.dot(s[:, np.newaxis]*vh)