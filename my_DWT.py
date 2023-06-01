import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.laplacian_pyramid import rowdec, rowdec2
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import rowint, rowint2
from my_LP import quantise
from cued_sf2_lab.jpeg import quant1, quant2
from common import *


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

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray, qrise=None, factors = None, strength = 0):
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
    if factors is None:
        factors = np.ones(dwtstep.shape)
    for i in range(n):
        m = m//2
        Yq[:m, m:2*m] = quant1(Y[:m, m:2*m], dwtstep[0, i], qrise*factors[0, i]*strength)
        dwtent[0, i] =   bpp(Yq[:m, m:2*m])*m*m
        Yq[m:2*m, :m] = quant1(Y[m:2*m, :m], dwtstep[1, i], qrise*factors[1, i]*strength)
        dwtent[1, i] =   bpp(Yq[m:2*m, :m])*m*m
        Yq[m:2*m, m:2*m] = quant1(Y[m:2*m, m:2*m], dwtstep[2, i], qrise*factors[2, i]*strength)
        dwtent[2, i] =   bpp(Yq[m:2*m, m:2*m])*m*m
    Yq[:m, :m,] = quant1(Y[:m, :m], dwtstep[0, n], qrise*factors[0, n]*strength)
    dwtent[0, n] = bpp(Yq[:m, :m])*m*m

    return Yq, dwtent, dwtstep, qrise, factors, strength

def quantdwt2(Y: np.ndarray, qrise=None, factors = None, strength = 0):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    ratios = get_ratios(Y, N, g1, g2)
    dwtstep = np.ones((3, N+1))*ratios*step
    _, n = dwtstep.shape
    n -= 1
    Yq = np.zeros(Y.shape)
    dwtent = np.zeros(dwtstep.shape)
    m = Y.shape[0]
    if factors is None:
        factors = np.ones(dwtstep.shape)
    for i in range(n):
        m = m//2
        Yq[:m, m:2*m] = quant2(Y[:m, m:2*m], dwtstep[0, i], qrise*factors[0, i]*strength)
        dwtent[0, i] =   bpp(Yq[:m, m:2*m])*m*m
        Yq[m:2*m, :m] = quant2(Y[m:2*m, :m], dwtstep[1, i], qrise*factors[1, i]*strength)
        dwtent[1, i] =   bpp(Yq[m:2*m, :m])*m*m
        Yq[m:2*m, m:2*m] = quant2(Y[m:2*m, m:2*m], dwtstep[2, i], qrise*factors[2, i]*strength)
        dwtent[2, i] =   bpp(Yq[m:2*m, m:2*m])*m*m
    Yq[:m, :m,] = quant2(Y[:m, :m], dwtstep[0, n], qrise*factors[0, n]*strength)
    dwtent[0, n] = bpp(Yq[:m, :m])*m*m

    return Yq, dwtent




def get_factors(Y, N):
    factors = np.ones((3, N+1))
    m = Y.shape[0]
    for i in range(N):
        m = m//2
        factors[0, i] = np.std(Y[:m, m:2*m])
        factors[1, i] = np.std(Y[m:2*m, :m])
        factors[2, i] = np.std(Y[m:2*m, m:2*m])
    factors[0, N] = np.std(Y[:m, :m])
    factors = factors[0, N]/factors

    return factors

def DWT_quant(X, N, h1, h2, g1, g2, step = None, emse = True, qrise=None, strength=0):
    Xq = quantise(X, 17, qrise)
    rms_ref = np.std(Xq-X)
    print("rms_ref:", rms_ref)
    Y = DWT(X, N, h1, h2)
    factors = get_factors(Y, N)
    if emse: ratios = get_ratios(X, N, g1, g2)
    else: ratios = np.ones((3, N+1))
    if step is None: step = step_size_optimiser(X, h1, h2, g1, g2, rms_ref, np.linspace(1, 15, 100), N, ratios, factors, emse, qrise, strength=0)
    print("step:", step)

    # ratios = np.ones(np.shape(ratios))
    # for i in range(N+1):
    #     ratios[:, i] *= 0.98**i
    ratios = get_ratios(X, N, g1, g2)
    dwtstep = np.ones((3, N+1))*ratios*step

    return quantdwt(Y, dwtstep, qrise, factors, strength)





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

def step_size_optimiser(X, h1, h2, g1, g2, target_rms, steps, N, ratios, factors, emse = True, qrise=None, strength=0):
    error_list = []
    for step in steps:
        Y = DWT(X, N, h1, h2)
        Yq, _, dwtstep, qrise, factors, strength = quantdwt(Y, step*ratios, qrise, factors, strength)
        Z = inverse_DWT(Yq, N, g1, g2)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]

def DWT_analysis(X, N, h1, h2, g1, g2, step = None, emse=True, plot=False, qrise=None, strength=0):
    Yq, dwtent = DWT_quant(X, N, h1, h2, g1, g2, step, emse, qrise, strength)
    Z = inverse_DWT(Yq, N, g1, g2)
    
    entropy = np.sum(dwtent)
    HXq = bpp(quantise(X, 17, qrise))*256*256
    CR = HXq/entropy
    print("CR:", CR)
    print("bits:", entropy)
    print("rms:", np.std(Z-X))

    if plot:
        fig, axs = plt.subplots(1, 2)
        plot_image(Z, ax=axs[0])
        plot_image(X, ax=axs[1])

    return Z