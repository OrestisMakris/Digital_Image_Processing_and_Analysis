import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2

from noise_utils import *

tiger_data = scipy.io.loadmat('tiger.mat')
tiger_original = tiger_data['tiger'] 

tiger = np.clip(tiger_original, 0, 1.0).astype(np.float32)

MAXval = 1.0


snr_db = 15
gaussian_noisy_tiger = add_gaussian_noise(tiger, snr_db)

# τα φίλτρα που θα χρησιμοποιήσουμε
gaussian_filtered_ma = moving_average_filter(gaussian_noisy_tiger)
gaussian_filtered_median = median_filter(gaussian_noisy_tiger)

#υπολογισμός PSNR για το θορυβώδες και φιλτραρισμένο σήμα μας

psnr_gaussian_ma = peak_signal_noise_ratio(tiger, gaussian_filtered_ma)
psnr_gaussian_median = peak_signal_noise_ratio(tiger, gaussian_filtered_median)

print(f"Gaussian Noise-MovingAverage SNR: {psnr_gaussian_ma:.4f} ΔB")
print(f"Gaussian Noise-Median Filter PSNR: {psnr_gaussian_median:.4f} dB")

# κρουστικό θόρυβος αλατιού και πιπέρι
amount = 0.2
sp_noisy_tiger = add_saltpepper_noise(tiger, amount)

sp_filtered_ma = moving_average_filter(sp_noisy_tiger)
sp_filtered_median = median_filter(sp_noisy_tiger)


psnr_sp_ma = peak_signal_noise_ratio(tiger, sp_filtered_ma)
psnr_sp_median = peak_signal_noise_ratio(tiger, sp_filtered_median)

print(f"Salt & Pepper Noise-Moving Average PSNR: {psnr_sp_ma:.4f} dB")
print(f"Salt & Pepper Noise-MedianFilter PSNR: {psnr_sp_median:.4f} dB")

combined_noisy_tiger_temp = add_gaussian_noise(tiger, snr_db)
combined_noisy_tiger = add_saltpepper_noise(combined_noisy_tiger_temp, amount)

filtered_ma_first = moving_average_filter(combined_noisy_tiger)
combined_filtered_ma_median = median_filter(filtered_ma_first)

filtered_median_first = median_filter(combined_noisy_tiger)
combined_filtered_median_ma = moving_average_filter(filtered_median_first)

psnr_combined_ma_median = peak_signal_noise_ratio(tiger, combined_filtered_ma_median)
psnr_combined_median_ma = peak_signal_noise_ratio(tiger, combined_filtered_median_ma)

print(f"Combined Noise-MA then Median PSNR: {psnr_combined_ma_median:.4f} dB")
print(f"Combined Noise-Median then MA PSNR: {psnr_combined_median_ma:.4f} dB")


plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.imshow(tiger, cmap='gray') 
plt.title('Original Image')
plt.axis('off')
plt.subplot(3,3,2)
plt.imshow(gaussian_noisy_tiger, cmap='gray')
plt.title(f'Gaussian Noisy (SNR {snr_db}dB)')
plt.axis('off')
plt.subplot(3, 3,3)
plt.imshow(gaussian_filtered_median, cmap='gray')
plt.title(f'Gaussian - Median Filtered\nPSNR: {psnr_gaussian_median:.4f}dB')
plt.axis('off')
plt.subplot(3, 3,4)
plt.imshow(sp_noisy_tiger, cmap='gray')
plt.title(f'S&P Noisy ({amount*100:.0f}%)')
plt.axis('off')
plt.subplot(3, 3,5)
plt.imshow(sp_filtered_median, cmap='gray')
plt.title(f'S&P - Median Filtered\nPSNR: {psnr_sp_median:.4f}dB')
plt.axis('off')
plt.subplot(3, 3,6)
plt.imshow(combined_noisy_tiger, cmap='gray')
plt.title('Combined Noisy Image')
plt.axis('off')
plt.subplot(3, 3,7)
plt.imshow(combined_filtered_ma_median, cmap='gray')
plt.title(f'Combined - MA then Median\nPSNR: {psnr_combined_ma_median:.4f}dB')
plt.axis('off')
plt.subplot(3, 3,8)
plt.imshow(combined_filtered_median_ma, cmap='gray')
plt.title(f'Combined - Median then MA\nPSNR: {psnr_combined_median_ma:.4f}dB')
plt.axis('off')
plt.subplot(3, 3,9)
plt.imshow(gaussian_filtered_ma, cmap='gray')
plt.title(f'Gaussian - MA Filtered\nPSNR: {psnr_gaussian_ma:.4f}dB')
plt.axis('off')
plt.tight_layout()
plt.show()