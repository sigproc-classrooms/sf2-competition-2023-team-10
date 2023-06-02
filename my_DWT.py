import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.laplacian_pyramid import rowdec, rowdec2
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import rowint, rowint2
from my_LP import quantise
from cued_sf2_lab.jpeg import quant1, quant2
from common import *
from cued_sf2_lab.jpeg import (dwtgroup, runampl, huffenc, 
        diagscan, huffdes, huffgen, huffdflt, quant2)


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

def quantdwt(Y: np.ndarray, dwtstep, factors, qrise=None, strength = 0):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    # TODO move this outside the functions as the ratios will always be the same

    _, n = dwtstep.shape
    n -= 1
    Yq = np.zeros(Y.shape)
    m = Y.shape[0]
    # factors = get_factors(Y, N)

    for i in range(n):
        m = m//2
        Yq[:m, m:2*m] = quant1(Y[:m, m:2*m], dwtstep[0, i], qrise*factors[0, i]*strength)
        Yq[m:2*m, :m] = quant1(Y[m:2*m, :m], dwtstep[1, i], qrise*factors[1, i]*strength)
        Yq[m:2*m, m:2*m] = quant1(Y[m:2*m, m:2*m], dwtstep[2, i], qrise*factors[2, i]*strength)
    Yq[:m, :m,] = quant1(Y[:m, :m], dwtstep[0, n], qrise*factors[0, n]*strength)

    return Yq, factors

def quantdwt2(Y: np.ndarray, step, factors, qrise=None, strength = 0):
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
    m = Y.shape[0]

    for i in range(n):
        m = m//2
        Yq[:m, m:2*m] = quant2(Y[:m, m:2*m], dwtstep[0, i], qrise*factors[0, i]*strength)
        Yq[m:2*m, :m] = quant2(Y[m:2*m, :m], dwtstep[1, i], qrise*factors[1, i]*strength)
        Yq[m:2*m, m:2*m] = quant2(Y[m:2*m, m:2*m], dwtstep[2, i], qrise*factors[2, i]*strength)
    Yq[:m, :m,] = quant2(Y[:m, :m], dwtstep[0, n], qrise*factors[0, n]*strength)

    return Yq




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

    Y = DWT(X, N, h1, h2)
    ratios = get_ratios(Y, N, g1, g2)
    factors = get_factors(Y, N)
    # Y, h1, h2, g1, g2, N, target_bits = 38500, emse = True, qrise=None, strength=0)
    if step is None: step = step_size_optimiser_new(Y, h1, h2, g1, g2, N, ratios, factors, 38500, emse, qrise, strength=strength)
    print("step:", step)
    dwtstep = np.ones((3, N+1))*ratios*step
    print("bananas")
    # ratios = np.ones(np.shape(ratios))
    # for i in range(N+1):
    #     ratios[:, i] *= 0.98**i
    Yq, factors = quantdwt(Y, dwtstep, factors, qrise, strength)
    print("apple sauce")
    return Yq, factors, step





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


def DWT_huffenc(Yq: np.ndarray, N: int = 8,
        opthuff: bool = False, dcbits = 8, log: bool = True
        ):

    Yq = Yq.astype('int')
    Yqr = dwtgroup(Yq, N)
    M = 2**N
    sy = Yqr.shape
    huffhist = np.zeros(16 ** 2)
    scan = diagscan(M)
    vlc = []
    dhufftab = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dhufftab)
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yqr = Yqr[r:r+M,c:c+M]
            
            yqrflat = yqr.flatten('F')
            # Encode DC coefficient first
            dccoef = yqrflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                raise ValueError(
                    'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqrflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, dhufftab

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yqr = Yqr[r:r+M, c:c+M]
            yqrflat = yqr.flatten('F')
            # Encode DC coefficient first
            dccoef = yqrflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqrflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape))*8))

    return vlc, dhufftab

def step_size_optimiser(X, h1, h2, g1, g2, target_rms, steps, N, ratios, factors, emse = True, qrise=None, strength=0):
    error_list = []
    Y = DWT(X, N, h1, h2)
    for step in steps:
        print(step)
        Yq, _ = quantdwt(Y, step*ratios, factors, qrise, strength)
        Z = inverse_DWT(Yq, N, g1, g2)
        error_list.append(np.abs(np.std(Z-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index]


def step_size_optimiser_new(Y, h1, h2, g1, g2, N, ratios, factors, target_bits = 38500,  emse = True, qrise=None, strength=0):
    # error_list = []
    def encoded_size(step):
        dwtstep = np.ones((3, N+1))*ratios*step
        Yq, _ = quantdwt(Y, dwtstep, factors, qrise, strength)
        vlc, header = DWT_huffenc(Yq, N, dcbits=12, opthuff=True, log=False)
        bits = np.sum(vlc[:, 1])
        print("bits: {}, step: {}".format(bits, step))
        return np.abs(bits-target_bits)

    return minimize_scalar(encoded_size, bounds=(1, 20)).x



def DWT_analysis(X, N, h1, h2, g1, g2, step = None, emse=True, plot=False, qrise=None, strength=0):
    Yq, _, __ = DWT_quant(X, N, h1, h2, g1, g2, step, emse, qrise, strength)
    Z = inverse_DWT(Yq, N, g1, g2)
    
    HXq = bpp(quantise(X, 17, qrise))*256*256
    # CR = HXq/entropy
    # print("CR:", CR)
    # print("bits:", entropy)
    print("rms:", np.std(Z-X))

    if plot:
        fig, axs = plt.subplots(1, 2)
        plot_image(Z, ax=axs[0])
        plot_image(X, ax=axs[1])

    return Z