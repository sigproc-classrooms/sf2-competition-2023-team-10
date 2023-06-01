import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import colxfm
from Daniel.my_DCT import DCT, inverse_DCT, dctbpp, inverse_regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import bpp
from Daniel.my_LP import quantise
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

def LBT_quant(X, N, s=1.33, qrise=None, strength = 0):
    X_q = quantise(X, 17, qrise)
    rms_ref = np.std(X-X_q)
    print("rms_ref:", rms_ref)
    step = step_size_optimiser(X, N, rms_ref, np.linspace(0, 40, 100), N, s, qrise, strength)
    print("step:", step)

    Y = LBT(X, N, s)

    return quantise_and_suppress(Y, N, step, qrise, strength)

def quantise_and_suppress(Y, N, step, qrise = None, strength = 0):
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

    return inverse_regroup(Yr_quant, N)


def step_size_optimiser(X, N, target_rms, range, n, s, qrise=None, strength = 0):
    steps = range
    error_list = []
    Y = LBT(X, n, s)
    for step in steps:
        Yq = quantise_and_suppress(Y, N, step, qrise, strength)
        Z = inverse_LBT(Yq, n, s)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]


def LBT_analysis(X, N, s=1.33, plot = False, qrise=None, suppression_strength = 0):
    
    Yq = LBT_quant(X, N, s, qrise, suppression_strength)
    Yr = regroup(Yq, N)/N

    Z = inverse_LBT(Yq, N, s)

    HXq = bpp(quantise(X, 17, qrise))*256*256
    bits, _ = dctbpp(Yr, 16)
    CR = HXq/bits
    print("CR:", CR)
    print("rms:", np.std(Z-X))

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax = ax)
    
    return Z