import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#read the image
img_path = os.path.join('Images', 'Ασκηση 1', 'moon.jpg')
img = plt.imread(img_path).astype(np.float64)

#normalize the image to 0,255 contrast stretching

min_val= img.min()
max_val= img.max()
img = (img - min_val) *(255.0/(max_val-min_val))

#convert to grayscale
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

#show the image
# plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.title('Original Image')
# plt.show()


# Get the dimensions of the image
M, N = img.shape

# Fourier fast transform

def dft1d(x):
    """
    Compute the 1D Discrete Fast Fourier Transform of the 1D array x.
    """
    N = x.shape[0]
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


#μεταφέρετε το συχνοτικό σημείο (0,0) στο κέντρο του πεδίου, με χρήση κατάλληλης ιδιότητας του μετασχηματισμού DFT
#DFT inrow
dft_rows = np.zeros((M, N), dtype=complex)
for m in range(M):
    dft_rows[m, :] = dft1d(img[m, :])

#DFT in column
dft_cols = np.zeros((M, N), dtype=complex)
for n in range(N):
    dft_cols[:, n] = dft1d(dft_rows[:, n])

#Zero frequency in the center
F_shifted = np.fft.fftshift(dft_cols)

#show the magnitude spectrum
magnitude_spectrum_lin = np.abs(F_shifted)
magnitude_spectrum_log= np.log(1 + magnitude_spectrum_lin)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(magnitude_spectrum_lin, cmap='gray')
plt.title('Amplitude (Linear)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum_log, cmap='gray')
plt.title('Amplitude (Log)')
plt.axis('off')
plt.show()

# low pass filter in the frequency domain
#radius of the filter
radius_filter = 400
# create a low pass filter
# calculate the distance from the center of the image
u0 = M // 2
v0 = N // 2

# low pass filter
def low_pass_filter(X, radius, u0, v0):
    """
    Create a low pass filter.   
    """
    U, V = np.meshgrid(np.arange(X.shape[1]), np.arange(X.shape[0]))
    D = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
    H = np.zeros(X.shape, dtype=float)
    H[D <= radius] = 1
    return H


# create the low pass filter
H = low_pass_filter(F_shifted, radius_filter, u0, v0)
# apply the filter to the frequency domain
F_filtered = F_shifted * H

# inverse zero frequency shift
F_filtered = np.fft.ifftshift(F_filtered)

# show the filtered magnitude spectrum
magnitude_spectrum_filtered_lin = np.abs(F_filtered)
magnitude_spectrum_filtered_log= np.log(1 + magnitude_spectrum_filtered_lin)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(magnitude_spectrum_filtered_lin, cmap='gray')
plt.title('Filtered Amplitude (Linear)')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum_filtered_log, cmap='gray')
plt.title('Filtered Amplitude (Log)')



#idft in column
idft_cols = np.zeros_like(F_filtered, dtype=complex)
for n in range(N):
    idft_cols[:, n] = dft1d(F_filtered[:, n].conjugate()).conjugate()

#idft in row
idft_rows = np.zeros_like(idft_cols, dtype=complex)
for m in range(M):
    idft_rows[m, :] = dft1d(idft_cols[m, :].conjugate()).conjugate()

image_restored = np.real(idft_rows) 


#undo contrast stretching
image_restored = (image_restored *(max_val-min_val)/255.0) + min_val


# save the image
output_path = os.path.join('Images', 'Ασκηση 1', 'moon_restored.jpg')
plt.imsave(output_path, image_restored, cmap='gray')
# show the original and restored image
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(image_restored, cmap='gray')
plt.axis('off')
plt.title('Restored Image')
plt.show()
