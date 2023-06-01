""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc
from my_DWT import DWT_quant
from huffman import DWT_huffenc

from .common import HeaderType, jpeg_quant_size

def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # replace this with your size estimate, and a comment explaining how you got it!
    return (len(header.bits) + len(header.huffval)) * 8


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
    Yq, _, dwtstep, qrise, factors, strength = DWT_quant(X, N, h1, h2, g1, g2, qrise = qrise, strength=strength)

    return DWT_huffenc(Yq, N, dcbits=12, opthuff=True)
