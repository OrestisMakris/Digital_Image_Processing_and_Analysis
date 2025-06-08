import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import convolve2d
import cv2  # Using OpenCV

# Load the image
try:
    tiger = scipy.io.loadmat('tiger.mat')['tiger']
    # Assuming tiger image is in range 0-255, if not, normalize or adjust MAX_I in PSNR
    if tiger.max() <= 1.0: # If image is float in [0,1]
        MAX_I = 1.0
    else: # Else assume uint8 range
        MAX_I = 255.0
except FileNotFoundError:
    print("Error: tiger.mat not found. Please make sure the file is in the correct directory.")
    exit()
except KeyError:
    print("Error: 'tiger' key not found in tiger.mat.")
    exit()


def peak_signal_noise_ratio(original, compressed):
    """Computes the PSNR between two images.

    Args:
        original (np.ndarray): The original image.
        compressed (np.ndarray): The compressed or noisy image.

    Returns:
        float: The PSNR value in dB.
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have infinite value.
        return float('inf')
    
    global MAX_I # Use the globally determined MAX_I
    psnr = 20 * np.log10(MAX_I / np.sqrt(mse))
    return psnr

def add_gaussian_noise(image, snr_db):
    """Adds Gaussian noise to an image to achieve a specified SNR.

    Args:
        image (np.ndarray): The input image.
        snr_db (float): The desired signal-to-noise ratio in dB.

    Returns:
        np.ndarray: The noisy image.
    """
    # Calculate signal power
    signal_power = np.mean(image ** 2)
    # Calculate noise power
    noise_power = signal_power / (10 ** (snr_db / 10))
    # Calculate noise standard deviation
    noise_std = np.sqrt(noise_power)
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    # Add noise to the image
    noisy_image = image + noise
    return noisy_image

def add_salt_and_pepper_noise(image, amount):
    """Adds salt and pepper noise to an image.

    Args:
        image (np.ndarray): The input image.
        amount (float): The proportion of image pixels to replace with noise.

    Returns:
        np.ndarray: The noisy image.
    """
    noisy_image = image.copy()
    # Salt noise
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = MAX_I # Use MAX_I for salt

    # Pepper noise
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

def moving_average_filter(image, kernel_size=3):
    """Applies a moving average filter to an image.

    Args:
        image (np.ndarray): The input image.
        kernel_size (int): The size of the moving average kernel (default is 3).

    Returns:
        np.ndarray: The filtered image.
    """
    # OpenCV's blur function is a moving average filter
    filtered_image = cv2.blur(image.astype(np.float32), (kernel_size, kernel_size))
    return filtered_image

def median_filter(image, kernel_size=3):
    """Applies a median filter to an image using OpenCV.

    Args:
        image (np.ndarray): The input image.
        kernel_size (int): The size of the median filter kernel (default is 3).
                           Must be an odd integer.
    Returns:
        np.ndarray: The filtered image.
    """
    # Ensure image is in a format OpenCV medianBlur can handle, e.g., uint8 or float32
    if image.dtype != np.uint8 and image.dtype != np.float32:
        # Attempt to convert to float32, assuming values are somewhat normalized
        # If original image was 0-255, this conversion is fine.
        # If it was 0-1, it's also fine.
        image_for_median = image.astype(np.float32)
    else:
        image_for_median = image

    # If image was float and MAX_I is 255, it might need scaling for uint8 conversion
    # However, cv2.medianBlur handles float32 directly.
    # For uint8, values must be in 0-255.
    # Let's assume if it's not uint8, it's float and can be used directly or converted.
    if image_for_median.max() > 1.0 and image_for_median.dtype == np.float32 and MAX_I == 255.0 :
         # If float but seems to be in 0-255 range, convert to uint8 for medianBlur if needed
         # or ensure it's correctly scaled if it was intended to be 0-1.
         # For simplicity, if it's float32, cv2.medianBlur should handle it.
         # If it's > 1 and not uint8, it might be an issue for some cv2 functions,
         # but medianBlur is generally robust.
         pass


    # Median blur requires kernel_size to be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        print(f"Median filter kernel size must be odd, changed to {kernel_size}")

    # Convert to uint8 if the image is in 0-255 range for cv2.medianBlur if it's not float32
    # If the image is already float32, cv2.medianBlur can handle it.
    if image_for_median.dtype != np.float32 and image_for_median.dtype != np.uint8:
        if MAX_I == 255.0: # Assuming it's a 0-255 range image stored as another type
            image_for_median = np.clip(image_for_median, 0, 255).astype(np.uint8)
        # else, if MAX_I is 1.0, it should ideally be float32.
        # This part can be tricky if the input image type isn't standard.

    # If tiger image was loaded as float64 by scipy.io.loadmat and is in 0-255 range
    if tiger.dtype == np.float64 and MAX_I == 255.0:
        image_for_median = image.astype(np.uint8)


    filtered_image = cv2.medianBlur(image_for_median, kernel_size)
    return filtered_image

# 1. Gaussian Noise
snr_db = 15
gaussian_noisy_tiger = add_gaussian_noise(tiger.astype(np.float32), snr_db) # Ensure float for noise addition

# Apply filters
gaussian_filtered_ma = moving_average_filter(gaussian_noisy_tiger)
gaussian_filtered_median = median_filter(gaussian_noisy_tiger)

# Calculate PSNR
psnr_gaussian_ma = peak_signal_noise_ratio(tiger, gaussian_filtered_ma)
psnr_gaussian_median = peak_signal_noise_ratio(tiger, gaussian_filtered_median)

print(f"Gaussian Noise - Moving Average PSNR: {psnr_gaussian_ma:.2f} dB")
print(f"Gaussian Noise - Median Filter PSNR: {psnr_gaussian_median:.2f} dB")

# 2. Salt and Pepper Noise
amount = 0.2
# Ensure tiger is in a suitable format for add_salt_and_pepper_noise if it modifies based on MAX_I
sp_noisy_tiger = add_salt_and_pepper_noise(tiger.copy(), amount)

# Apply filters
sp_filtered_ma = moving_average_filter(sp_noisy_tiger)
sp_filtered_median = median_filter(sp_noisy_tiger)

# Calculate PSNR
psnr_sp_ma = peak_signal_noise_ratio(tiger, sp_filtered_ma)
psnr_sp_median = peak_signal_noise_ratio(tiger, sp_filtered_median)

print(f"Salt & Pepper Noise - Moving Average PSNR: {psnr_sp_ma:.2f} dB")
print(f"Salt & Pepper Noise - Median Filter PSNR: {psnr_sp_median:.2f} dB")

# 3. Combined Noise
# Start with a float version for Gaussian noise, then S&P might convert to uint8 if MAX_I is 255
combined_noisy_tiger_float = add_gaussian_noise(tiger.astype(np.float32), snr_db)
combined_noisy_tiger = add_salt_and_pepper_noise(combined_noisy_tiger_float, amount)


# Apply filters in sequence
combined_filtered_ma_median = median_filter(moving_average_filter(combined_noisy_tiger))
combined_filtered_median_ma = moving_average_filter(median_filter(combined_noisy_tiger))

# Calculate PSNR
psnr_combined_ma_median = peak_signal_noise_ratio(tiger, combined_filtered_ma_median)
psnr_combined_median_ma = peak_signal_noise_ratio(tiger, combined_filtered_median_ma)

print(f"Combined Noise - MA then Median PSNR: {psnr_combined_ma_median:.2f} dB")
print(f"Combined Noise - Median then MA PSNR: {psnr_combined_median_ma:.2f} dB")

# Display Results (Optional)
plt.figure(figsize=(15, 12)) # Adjusted figure size for better layout

plt.subplot(3, 3, 1)
plt.imshow(tiger, cmap='gray', vmin=0, vmax=MAX_I)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(gaussian_noisy_tiger, cmap='gray', vmin=0, vmax=MAX_I if MAX_I == 255.0 else None)
plt.title(f'Gaussian Noisy (SNR {snr_db}dB)')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(gaussian_filtered_median, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'Gaussian - Median Filtered\nPSNR: {psnr_gaussian_median:.2f}dB')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(sp_noisy_tiger, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'S&P Noisy ({amount*100:.0f}%)')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(sp_filtered_median, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'S&P - Median Filtered\nPSNR: {psnr_sp_median:.2f}dB')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(combined_noisy_tiger, cmap='gray', vmin=0, vmax=MAX_I if MAX_I == 255.0 else None)
plt.title('Combined Noisy Image')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(combined_filtered_ma_median, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'Combined - MA then Median\nPSNR: {psnr_combined_ma_median:.2f}dB')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(combined_filtered_median_ma, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'Combined - Median then MA\nPSNR: {psnr_combined_median_ma:.2f}dB')
plt.axis('off')

# Placeholder for the 9th subplot if needed, or remove it
plt.subplot(3, 3, 9)
# Example: Show MA filtered Gaussian for comparison
plt.imshow(gaussian_filtered_ma, cmap='gray', vmin=0, vmax=MAX_I)
plt.title(f'Gaussian - MA Filtered\nPSNR: {psnr_gaussian_ma:.2f}dB')
plt.axis('off')


plt.tight_layout()
plt.show()
