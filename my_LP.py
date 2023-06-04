from cued_sf2_lab.simple_image_filtering import halfcos
from cued_sf2_lab.laplacian_pyramid import rowdec, rowint
from cued_sf2_lab.laplacian_pyramid import bpp, quant1, quant2
from cued_sf2_lab.familiarisation import plot_image
import numpy as np
import matplotlib.pyplot as plt


def quantise(x, step, rise1=None):
    """
    Quantise matrix x in one go with step width of step using quant1 and quant2

    If rise1 is defined, the first step rises at rise1, otherwise it rises at
    step/2 to give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = x.copy()
        return y
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1*step
    # Perform both quantisation steps
    y = quant2(quant1(x, step, rise), step, rise)
    return y


def halfcos_filter(n):
    return halfcos(n)


def LP(X, h, N):
    Y_list = []
    X_prev = X
    for i in range(N):
        X_downsampled = rowdec(rowdec(X_prev.T, h).T, h)
        X_upscaled = rowint(rowint(X_downsampled.T, 2*h).T, 2*h)
        Y_list.append(X_prev-X_upscaled)
        X_prev = X_downsampled
    return *Y_list, X_prev


def inverse_LP(pyramid, h):
    Z_list = []
    Z_last = pyramid[-1]
    for i in range(2, len(pyramid)+1):
        Z_last = rowint(rowint(Z_last.T, 2*h).T, 2*h) + pyramid[-i]
        Z_list.append(Z_last)
    return Z_list


def get_ratios(X, N, h):
    X0 = np.zeros(X.shape)
    pyramid = LP(X0, h, N)
    ssim_values = []
    ratios = []
    for layer in range(N+1):
        pos = len(pyramid[layer])//2
        pyramid[layer][pos, pos] = 100

        decoded = inverse_LP(pyramid, h)
        ssim_values.append(np.sum(np.abs(decoded[-1]**2.0)))
    for energy in ssim_values:
        ratios.append(np.sqrt(ssim_values[-1]/energy))
    return ratios


def step_size_optimiser(X, h, target_rms, range, N, equal_mse=True, qrise=None):
    if equal_mse:
        ratios = get_ratios(X, N, h)
    else:
        ratios = np.ones(N+1)
    steps = range
    error_list = []
    pyramid = LP(X, h, N)
    for step in steps:
        pyramid_q = [quantise(image, step*ratios[i], qrise)
                     for i, image in enumerate(pyramid)]
        decoded = inverse_LP(pyramid_q, h)
        error_list.append(np.abs(np.std(decoded[-1]-X)-target_rms))
    min_index = error_list.index(min(error_list))
    return steps[min_index], ratios


def LP_quant(X, N, h, equal_mse=True, qrise=None):
    X_q = quantise(X, 17, qrise)
    ref_rms = np.std(X-X_q)
    s, ratios = step_size_optimiser(
        X, h, ref_rms, np.linspace(0, 12, 100), N, equal_mse, qrise)
    steps = [s*r for r in ratios]
    print("steps:", steps)

    pyramid = LP(X, h, N)
    return [quantise(image, s*ratios[i], qrise) for i, image in enumerate(pyramid)]


def LP_anaysis(X, N, h, equal_mse=True, plot=False, qrise=None):
    pyramid_q = LP_quant(X, N, h, equal_mse, qrise)
    decoded = inverse_LP(pyramid_q, h)

    if plot:
        fig, ax = plt.subplots()
        plot_image(decoded[-1], ax=ax)

    HXq = bpp(quantise(X, 17, qrise))*256*256
    entropy = 0
    for image in pyramid_q:
        entropy += bpp(image)*image.shape[0]*image.shape[1]

    print("ratio:", HXq/entropy)
    return decoded[-1]
