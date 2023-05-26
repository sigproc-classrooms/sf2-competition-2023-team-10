import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import bpp, quantise

def dctbpp(Yr, N):
    bits = 0
    size = Yr.shape[0]//N
    for i in range(N):
        for j in range(N):
            subimage = Yr[i*size:(i+1)*size, j*size:(j+1)*size]
            bits += bpp(subimage)*size*size
    return bits

def DCT(X, n):
    Cn = dct_ii(n)
    return colxfm(colxfm(X, Cn).T, Cn).T

def DCT_quant(X, N, qrise = None):
    X_q = quantise(X, 17, qrise)
    rms_ref = np.std(X-X_q)
    step = step_size_optimiser(X, rms_ref, np.linspace(0, 40, 100), N, qrise)
    print("step:", step)

    Y = DCT(X, N)
    return quantise(Y, step, qrise)

def inverse_DCT(Y, n):
    Cn = dct_ii(n)
    return colxfm(colxfm(Y.T, Cn.T).T, Cn.T)

def step_size_optimiser(X, target_rms, range, n, qrise=None):
    steps = range
    error_list = []
    Y = DCT(X, n)
    for step in steps:
        Yq = quantise(Y, step, qrise)
        Z = inverse_DCT(Yq, n)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]

def DCT_analysis(X, N, plot = False, qrise=None):
    
    Yq = DCT_quant(X, N, qrise)
    Yr = regroup(Yq, N)/N

    Z = inverse_DCT(Yq, N)

    HXq = bpp(quantise(X, 17, qrise))*256*256
    CR = HXq/dctbpp(Yr, 8)
    print("CR:", CR)

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax = ax)

    return Z