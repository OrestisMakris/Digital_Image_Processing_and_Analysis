import numpy as np
import matplotlib.pyplot as plt
import os

def read_and_normalize(path):

    """
    Read an image from disk, convert t  float, stretch to 0,255
    Returns (img, min_val, max_val

    """

    img = plt.imread(path).astype(np.float64)
    mn, mx = img.min(), img.max()
    norm = (img - mn)*(255.0/(mx - mn))

    return norm, mn, mx

def rgb2gray(img):

    """
    If RGB, convert t
    o grayscale using luminance weights.
    """

    if img.ndim == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img

def dft1d(x):

    """
    Compute 1D DFT of x (naïve O(N^2) implementation).
    x is a 1D numpy array.
    """

    N = x.shape[0]
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n]*np.exp(-2j*np.pi*k*n/N)
    return X

def dft2d(img):

    """
    Compute 2D DFT by successive 1D transforms on rows then columns
    """

    M, N = img.shape
    tmp = np.zeros((M, N), dtype=complex)

    for m in range(M):
        tmp[m, :] = dft1d(img[m, :])
    out = np.zeros_like(tmp)

    for n in range(N):
        out[:, n] = dft1d(tmp[:, n])
    return out

def idft2d(F):

    """
    Inverse 2D DFT (conjugate trick + 1D iDFT)
    """

    M, N = F.shape
    tmp = np.zeros_like(F, dtype=complex)

    for n in range(N):
        tmp[:, n] = dft1d(F[:, n].conjugate()).conjugate()
    out = np.zeros_like(tmp, dtype=complex)

    for m in range(M):
        out[m, :] = dft1d(tmp[m, :].conjugate()).conjugate()
    return np.real(out)

def fftshift(F):
   
    """Shift zero-frequency component to the center of the spectrum"""

    return np.fft.fftshift(F)

def ifftshift(F):

    """Invers shift of zero-frequency component"""

    return np.fft.ifftshift(F)

def low_pass_filter(shape, radius, center):

    """
    Create a binary circular low-pass mask of given radius
     shape=(rows,cols), center=(u0,v0)

    """
    M, N = shape
    U, V = np.meshgrid(np.arange(N), np.arange(M))

    D = np.sqrt((U-center[1])**2 + (V-center[0])**2)
    H = np.zeros(shape, dtype=float)
    #τιμεσ 0-1
    H[D <= radius] = 1.0

    return H

def plot_spectrum(F, log=False, cmap='gray', title=None):

    """
    Plot magnitude spectrum. Set log=True for log scale.
    F is a 2D complex array (e.g., DFT result)
    """
    mag = np.abs(F)

    if log:
        mag = np.log1p(mag)
    plt.imshow(mag, cmap=cmap)

    if title:
        plt.title(title)
    plt.axis('off')