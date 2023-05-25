""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc

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
    vlc, hufftab = jpegenc(X, jpeg_quant_size, opthuff=True, log=False)
    a, b = (1, 2, 3)
    return vlc, hufftab
