import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from Daniel.my_LP import quantise

def dctbpp(Yr, N):
    bits = 0
    size = Yr.shape[0]//N
    entropies = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            subimage = Yr[i*size:(i+1)*size, j*size:(j+1)*size]
            bits += bpp(subimage)*size*size
            entropies[i, j] = np.std(subimage)
    return bits, entropies


def inverse_regroup(Yr, N):
    Y = np.zeros(Yr.shape)
    m, n = Yr.shape
    for i in range(N):
        for j in range(N):
            Y[i::N, j::N] = Yr[i*m//N:(i+1)*m//N, j*n//N:(j+1)*n//N]
    return Y

def DCT(X, n):
    Cn = dct_ii(n)
    return colxfm(colxfm(X, Cn).T, Cn).T


def quantise_and_suppress(Y, N, step, qrise=None, strength=0):
    Yr = regroup(Y, N)
    m, n = Yr.shape
    m = m//N
    n = n//N
    Yr_quant = np.zeros(Yr.shape)
    _, entropies = dctbpp(Yr, N)
    factors = entropies[0, 0]/entropies
    for i in range(N):
        for j in range(N):
            Yr_quant[i*m:(i+1)*m, j*n:(j+1)*n] = quantise(Yr[i*m:(i+1)*m, j*n:(j+1)*n], step, qrise*np.exp((factors[i, j]-1)*strength))

    return inverse_regroup(Yr_quant, 8)

def DCT_quant(X, N, qrise = None, strength = 0):
    X_q = quantise(X, 17, qrise)
    rms_ref = np.std(X-X_q)
    print("rms_ref:", rms_ref)
    step = step_size_optimiser(X, N, rms_ref, np.linspace(0, 40, 100), qrise, strength)
    print("step:", step)

    Y = DCT(X, N)

    return quantise_and_suppress(Y, N, step, qrise, strength)

def inverse_DCT(Y, n):
    Cn = dct_ii(n)
    return colxfm(colxfm(Y.T, Cn.T).T, Cn.T)

def step_size_optimiser(X, N, target_rms, range, qrise=None, strength=None):
    steps = range
    error_list = []
    Y = DCT(X, N)
    for step in steps:
        Yq = quantise_and_suppress(Y, N, step, qrise, strength)
        Z = inverse_DCT(Yq, N)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]

def DCT_analysis(X, N, plot = False, qrise=None, suppression_strength = 0):
    
    Yq = DCT_quant(X, N, qrise, strength=suppression_strength)
    Yr = regroup(Yq, N)/N

    Z = inverse_DCT(Yq, N)

    HXq = bpp(quantise(X, 17, qrise))*256*256
    bits, _ = dctbpp(Yr, 8)
    CR = HXq/bits
    print("CR:", CR)
    print("rms:", np.std(Z-X))

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax = ax)

    return Z