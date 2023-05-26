import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import colxfm
from Daniel.my_DCT import DCT, inverse_DCT
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dct import regroup



def LBT(X, N, s=1.33):
    Pf, Pr = pot_ii(N, s)

    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    return DCT(Xp, N)

def inverse_LBT(Y, N, s=1.33):
    Pf, Pr = pot_ii(N, s)

    Z  =  inverse_DCT(Y, N) 

    t = np.s_[N//2:-N//2]
    Zp = Z.copy()  #copy the non-transformed edges directly from Z
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)

    return Zp

def LBT_quant(X, N, s=1.33, qrise=None):
    X_q = quantise(X, 17, qrise)
    rms_ref = np.std(X-X_q)
    step = step_size_optimiser(X, rms_ref, np.linspace(0, 40, 100), N, s, qrise)
    print("step:", step)

    Y = LBT(X, N, s)
    return quantise(Y, step, qrise)


def step_size_optimiser(X, target_rms, range, n, s, qrise=None):
    steps = range
    error_list = []
    Y = LBT(X, n, s)
    for step in steps:
        Yq = quantise(Y, step, qrise)
        Z = inverse_LBT(Yq, n, s)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]


def dctbpp(Yr, N):
    # Your code here
    bits = 0
    size = Yr.shape[0]//N
    for i in range(N):
        for j in range(N):
            subimage = Yr[i*size:(i+1)*size, j*size:(j+1)*size]
            bits += bpp(subimage)*size*size
    return bits

def LBT_analysis(X, N, s=1.33, plot = False, qrise=None):
    
    Yq = LBT_quant(X, N, s, qrise)
    Yr = regroup(Yq, N)/N

    Z = inverse_LBT(Yq, N, s)

    HXq = bpp(quantise(X, 17, qrise))*256*256
    CR = HXq/dctbpp(Yr, 16)
    # CRn = bpp(X)*256*256/dctbpp(Yr, 16)
    print(CR)

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax = ax)