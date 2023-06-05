# put any code shared between your encoder and decoder here
from typing import Any
from cued_sf2_lab.jpeg import HuffmanTable
import numpy as np

# Here we indicate what format we plan to store our header in. This is just documentation for the
# demonstrator marking it, neither Python nor the competition runner care
HeaderType = HuffmanTable  # using the example jpeg encoder
# HeaderType = None        # when not using a header
# HeaderType = np.ndarray  # when your header is single array
# HeaderType = Any         # if you're using some other header forma

# a constant shared by the encoder and decoder
jpeg_quant_size = 128

N = 5
h1 = np.array([-1, 2, 6, 2, -1])/8
h2 = np.array([-1, 2, -1])/4
g1 = np.array([1, 2, 1])/2
g2 = np.array([-1, -2, 6, -2, -1])/4

step = 17

PCA_components = 94
svd_dimension = 100

def my_function():
    pass
