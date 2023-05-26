import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import plot_image
import numpy as np
from cued_sf2_lab.laplacian_pyramid import rowdec, rowdec2
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.laplacian_pyramid import rowint, rowint2


def DWT(X, N, h1, h2):
    m = X.shape[0]//2
    if N == 0: return X
    Y = np.concatenate([rowdec(X, h1), rowdec2(X, h2)], axis=1)
    Y = np.concatenate([rowdec(Y.T, h1).T, rowdec2(Y.T, h2).T], axis=0)
    Y[:m,:m] = DWT(Y[:m,:m], N-1, h1, h2)
    return Y

def inverse_DWT(X, N, g1, g2):
    m = X.shape[0]//2
    if N == 1: 
        Y = rowint(X[:m, :].T, g1).T + rowint2(X[m:, :].T,g2).T
        Y = rowint(Y[:, :m], g1) + rowint2(Y[:, m:], g2)
        return Y
    return inverse_DWT(np.block([[inverse_DWT(X[:m, :m], N-1, g1, g2), X[:m, m:]], [X[m:, :m], X[m:, m:]]]), 1, g1, g2)

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray, qrise=None):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    _, n = dwtstep.shape
    n -= 1
    Yq = np.zeros(Y.shape)
    dwtent = np.zeros(dwtstep.shape)
    m = Y.shape[0]
    for i in range(n):
        m = m//2
        Yq[:m, m:2*m] = quantise(Y[:m, m:2*m], dwtstep[0, i], qrise)
        dwtent[0, i] =   bpp(Yq[:m, m:2*m])*m*m
        Yq[m:2*m, :m] = quantise(Y[m:2*m, :m], dwtstep[1, i], qrise)
        dwtent[1, i] =   bpp(Yq[m:2*m, :m])*m*m
        Yq[m:2*m, m:2*m] = quantise(Y[m:2*m, m:2*m], dwtstep[2, i], qrise)
        dwtent[2, i] =   bpp(Yq[m:2*m, m:2*m])*m*m
    Yq[:m, :m,] = quantise(Y[:m, :m], dwtstep[0, n], qrise)
    dwtent[0, n] = bpp(Yq[:m, :m])*m*m

    return Yq, dwtent

def DWT_quant(X, N, h1, h2, g1, g2, emse = True, qrise=None):
    Xq = quantise(X, 17, qrise)
    rms_ref = np.std(Xq-X)
    step, ratios = step_size_optimiser(X, h1, h2, g1, g2, rms_ref, np.linspace(1, 15, 100), N, emse, qrise)
    print("step:", step)
    dwtstep = np.ones((3, N+1))*ratios*step
    Y = DWT(X, N, h1, h2)
    return quantdwt(Y, dwtstep, qrise)

def get_ratios(X, N, g1, g2):
    Y = np.zeros(X.shape)
    energies = np.zeros((3, N+1))
    m = X.shape[0]
    for i in range(N):
        m = m//2
        Y[m//2, 3*m//2] = 100
        Z = inverse_DWT(Y, N, g1, g2)
        energies[0, i] = np.sum(Z**2.0)
        Y*=0
        Y[3*m//2, m//2] = 100
        Z = inverse_DWT(Y, N, g1, g2)
        energies[1, i] = np.sum(Z**2.0)
        Y*=0
        Y[3*m//2, 3*m//2] = 100
        Z = inverse_DWT(Y, N, g1, g2)
        energies[2, i] = np.sum(Z**2.0)
        Y*=0
    Y[m//2, m//2] = 100
    Z = inverse_DWT(Y, N, g1, g2)
    energies[0, N] = np.sum(Z**2.0)
    energies[1, N] =1e10
    energies[2, N] =1e10
    # print(energies)
    return np.sqrt(energies[0, 0]/energies[:, :])

def step_size_optimiser(X, h1, h2, g1, g2, target_rms, steps, N, emse = True, qrise=None):
    if emse: ratios = get_ratios(X, N, g1, g2)
    else: ratios = np.ones((3, N+1))
    error_list = []
    for step in steps:
        Y = DWT(X, N, h1, h2)
        Yq, _ = quantdwt(Y, step*ratios, qrise)
        Z = inverse_DWT(Yq, N, g1, g2)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index], ratios

def DWT_analysis(X, N, h1, h2, g1, g2, emse=True, plot=False, qrise=None):
    Yq, dwtent = DWT_quant(X, N, h1, h2, g1, g2, emse, qrise)
    Z = inverse_DWT(Yq, N, g1, g2)
    
    entropy = np.sum(dwtent)
    HXq = bpp(quantise(X, 17, qrise))*256*256
    CR = HXq/entropy
    print("CR:", CR)

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)