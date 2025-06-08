import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    read_and_normalize, rgb2gray,
    dft2d, idft2d,
    fftshift, ifftshift,
    low_pass_filter, plot_spectrum
)

# Read  normalize grayscale
img_path = os.path.join('moon.jpg')
img, mn, mx = read_and_normalize(img_path)
img = rgb2gray(img)

# 2D DFT  center
F = fftshift(dft2d(img))

# ΠΛΟΤ  original spectrum
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_spectrum(F, log=False, title='Linear')
plt.subplot(1,2,2)
plot_spectrum(F, log=True,  title='Log')
plt.show()

# Φιλτρο χαμηλής συχνότιτας
M, N = img.shape
radius = 100
center = (M//2, N//2)

H = low_pass_filter(img.shape, radius=radius, center=center)

Ff = F * H

#  Unshift  inverse DFT

Ff = ifftshift(Ff)
restored = idft2d(Ff)


restored = restored*(mx-mn)/255.0 + mn

out = os.path.join('moon_restored.jpg')
#αποθήκευσι
plt.imsave(out, restored, cmap='gray')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img, cmap='gray');  plt.axis('off'); plt.title('Orig')
plt.subplot(1,2,2); plt.imshow(restored, cmap='gray'); plt.axis('off'); plt.title('Restored')
plt.show()