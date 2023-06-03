""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from .common import my_function, HeaderType, jpeg_quant_size
from my_DWT import *
from PCA_DWT import *
from huffman import *

def decode(vlc: np.ndarray, header: HeaderType) -> np.ndarray:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen decoding scheme

    header_huff, pca_object, factors, strength = header

    pca_decoded = PCA_huffdec(vlc, pca_object, hufftab=header_huff)

    DWT_decoded = inverse_PCA_DWT(pca_object, pca_decoded)

    DWT_final = quantdwt2(DWT_decoded, factors, strength=strength)
    reconstructed = inverse_DWT(DWT_final, N, g1, g2)


    return reconstructed
