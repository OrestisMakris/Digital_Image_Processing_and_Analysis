import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the image
img = plt.imread('image.jpg').astype(np.float64)

#normalize the image to 0,255 contrast stretching

min_val= img.min()
max_val= img.max()
img = (img - min_val) *(255.0/(max_val-min_val))


