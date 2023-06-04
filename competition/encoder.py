""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc
from my_DWT import DWT_quant
from huffman import *
from PCA_DWT import *

from .common import HeaderType, jpeg_quant_size

def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # replace this with your size estimate, and a comment explaining how you got it!
    header_huff, factors, strength = header

    factors_size = len(factors.flatten()) * 16
    strength_size = 4


    return (len(header_huff.bits) + len(header_huff.huffval)) * 8 + strength_size + factors_size


def encode(X: np.ndarray) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen encoding scheme. If you do not use a header,
    # then `return vlc, None`.
    

    DWT_result, factors, strength = DWT_quant(X, log = False)
    vlc, header_huff = DWT_huffenc(DWT_result, dcbits=12, opthuff=True)
    header = (header_huff, factors, strength)
    return vlc, header
