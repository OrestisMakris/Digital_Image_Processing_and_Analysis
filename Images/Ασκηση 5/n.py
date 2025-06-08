import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pickle
import os
from scipy.io import loadmat
from scipy.signal import convolve2d

# --- Helper Functions for MSE and PSNR ---
def mse(img1_float, img2_float):
    """Computes Mean Squared Error between two float images."""
    return np.mean((img1_float - img2_float) ** 2)

def psnr(img1_float, img2_float, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio between two float images."""
    err = mse(img1_float, img2_float)
    if err == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(err))

def main():
    # --- Common: Load Original Image ---
    try:
        img_ny_bgr = cv2.imread('new_york.png')
        if img_ny_bgr is None:
            raise FileNotFoundError("new_york.png not found. Please check the path.")
        img_ny_gray = cv2.cvtColor(img_ny_bgr, cv2.COLOR_BGR2GRAY)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading or converting new_york.png: {e}")
        return

    img_ny_float = img_ny_gray.astype(np.float64) / 255.0
    rows, cols = img_ny_float.shape

    # --- Part A: Wiener Filter for Denoising ---
    print("--- Part A: Wiener Filter for Denoising ---")
    snr_db_a = 10.0

    # Add Gaussian noise
    signal_power_a = np.mean(img_ny_float ** 2)
    snr_linear_a = 10 ** (snr_db_a / 10.0)
    noise_variance_a = signal_power_a / snr_linear_a # This is noise power (variance)
    noise_std_dev_a = np.sqrt(noise_variance_a)

    np.random.seed(0) # for reproducibility
    gaussian_noise_a = np.random.normal(0, noise_std_dev_a, img_ny_float.shape)
    noisy_img_a_float = img_ny_float + gaussian_noise_a
    noisy_img_a_clipped = np.clip(noisy_img_a_float, 0, 1)

    psnr_noisy_a = psnr(img_ny_float, noisy_img_a_clipped)
    print(f"Noisy image for Part A created. Target SNR: {snr_db_a}dB. Actual PSNR vs Original: {psnr_noisy_a:.2f}dB")

    # 1. Wiener filter with known noise power
    wiener_known_noise = wiener(noisy_img_a_clipped, mysize=(5,5), noise=noise_variance_a)
    wiener_known_noise_clipped = np.clip(wiener_known_noise, 0, 1)
    psnr_wiener_known = psnr(img_ny_float, wiener_known_noise_clipped)

    # 2. Wiener filter with unknown noise power (noise=None)
    wiener_unknown_noise = wiener(noisy_img_a_clipped, mysize=(5,5), noise=None)
    wiener_unknown_noise_clipped = np.clip(wiener_unknown_noise, 0, 1)
    psnr_wiener_unknown = psnr(img_ny_float, wiener_unknown_noise_clipped)

    # Display results for Part A
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 4, 1); plt.imshow(img_ny_float, cmap='gray', vmin=0, vmax=1); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(noisy_img_a_clipped, cmap='gray', vmin=0, vmax=1); plt.title(f'Noisy (PSNR: {psnr_noisy_a:.2f}dB)'); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(wiener_known_noise_clipped, cmap='gray', vmin=0, vmax=1); plt.title(f'Wiener (Known Noise)\nPSNR: {psnr_wiener_known:.2f}dB'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(wiener_unknown_noise_clipped, cmap='gray', vmin=0, vmax=1); plt.title(f'Wiener (Unknown Noise)\nPSNR: {psnr_wiener_unknown:.2f}dB'); plt.axis('off')
    plt.suptitle('Part A: Wiener Filter Denoising Results')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("\nComments on Part A Results:")
    print(f"  PSNR - Original vs Noisy: {psnr_noisy_a:.2f} dB")
    print(f"  PSNR - Original vs Wiener (Known Noise Variance): {psnr_wiener_known:.2f} dB")
    print(f"  PSNR - Original vs Wiener (Unknown Noise Variance): {psnr_wiener_unknown:.2f} dB")
    print("  - The Wiener filter, in both cases, improves the image quality by reducing noise, as indicated by the increase in PSNR.")
    print("  - Knowing the noise variance generally allows the Wiener filter to perform optimally, often yielding a slightly higher PSNR than when it has to estimate the noise characteristics locally from the image data.")
    print("  - Visually, both filtered images appear smoother and less noisy. The version with known noise variance might offer a better trade-off between noise reduction and preservation of fine image details.")
 # --- Part B: Inverse Filtering for Deblurring ---
    print("\n\n--- Part B: Inverse Filtering for Deblurring ---")

    psf_file = 'psf.p'
    if not os.path.exists(psf_file):
        print(f"Error: {psf_file} not found. This file is required for Part B.")
        return

    # 1) peek at first bytes
    with open(psf_file, 'rb') as _f:
        hdr = _f.read(32)
    print(f"psf.p header (first 32 bytes): {hdr!r}")

    psf_transform_function = None

    # 2) try dill
    try:
        import dill
        with open(psf_file, 'rb') as f:
            psf_transform_function = dill.load(f)
        print("Loaded psf.p via dill.")
    except Exception as e_dill:
        print(f"  dill.load failed: {e_dill}")

    # 3) try joblib
    if psf_transform_function is None:
        try:
            from joblib import load as jl_load
            psf_transform_function = jl_load(psf_file)
            print("Loaded psf.p via joblib.")
        except Exception as e_joblib:
            print(f"  joblib.load failed: {e_joblib}")

    # 4) fallback to scipy.io.loadmat
    if psf_transform_function is None:
        try:
            mat = loadmat(psf_file)
            print(f"Loaded as .mat, keys={mat.keys()}")
            for key in ('psf','h','PSF'):
                if key in mat:
                    psf_arr = mat[key]
                    def convolve_with_psf(img, psf=psf_arr):
                        return convolve2d(img, psf, mode='same', boundary='wrap')
                    psf_transform_function = convolve_with_psf
                    print(f"  using mat['{key}'] as PSF array")
                    break
        except Exception as e_mat:
            print(f"  loadmat failed: {e_mat}")

    if psf_transform_function is None:
        print("Could not load psf.p (dill, joblib or .mat).")
        return


    # Apply the PSF transform to the original clean image
    blurred_img_b_float = psf_transform_function(img_ny_float)
    blurred_img_b_clipped = np.clip(blurred_img_b_float, 0, 1)
    mse_blurred = mse(img_ny_float, blurred_img_b_clipped)
    psnr_blurred = psnr(img_ny_float, blurred_img_b_clipped)
    print(f"Image blurred using psf.p. MSE vs Original: {mse_blurred:.6f}, PSNR: {psnr_blurred:.2f}dB")

    # Estimate the impulse response (PSF) h
    delta_img = np.zeros_like(img_ny_float)
    delta_img[rows // 2, cols // 2] = 1.0 # Impulse at the center
    
    h_estimated = psf_transform_function(delta_img)
    # For FFT, the PSF's origin (center) should be at the (0,0) index.
    # If h_estimated is centered around the image center, use ifftshift.
    h_fft_input = ifftshift(h_estimated)

    # Frequency response H of the estimated PSF
    H_estimated = fft2(h_fft_input)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(fftshift(h_estimated), cmap='gray'); plt.title('Estimated PSF (h) (Center View)'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(np.log10(1 + np.abs(fftshift(H_estimated))), cmap='gray'); plt.title('Log Magnitude Spectrum of H (Center View)'); plt.axis('off')
    plt.suptitle('Part B: Estimated PSF and its Spectrum')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Inverse filtering with various thresholds
    G_blurred = fft2(blurred_img_b_clipped) # FFT of the blurred image
    
    threshold_values_inv = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2] # T=0 means no threshold
    mse_vs_threshold = []
    psnr_vs_threshold = []
    restored_image_cache = {}

    print("\nApplying Inverse Filter with different thresholds T for |H|:")
    for T_threshold in threshold_values_inv:
        abs_H = np.abs(H_estimated)
        H_inv_regularized = np.zeros_like(H_estimated, dtype=complex)
        
        if T_threshold == 0.0: # No thresholding (direct inverse)
            epsilon = 1e-9 # Avoid division by absolute zero if H has them
            H_inv_regularized = 1.0 / (H_estimated + epsilon)
        else:
            # Apply threshold: H_inv_regularized = 1/H where |H| > T, else 0
            mask = abs_H > T_threshold
            H_inv_regularized[mask] = 1.0 / H_estimated[mask]
            # Where abs_H <= T_threshold, H_inv_regularized remains 0.

        F_restored_freq = G_blurred * H_inv_regularized
        restored_img_ifft = np.real(ifft2(F_restored_freq))
        restored_img_clipped = np.clip(restored_img_ifft, 0, 1)
        
        restored_image_cache[T_threshold] = restored_img_clipped
        current_mse = mse(img_ny_float, restored_img_clipped)
        current_psnr = psnr(img_ny_float, restored_img_clipped)
        mse_vs_threshold.append(current_mse)
        psnr_vs_threshold.append(current_psnr)
        print(f"  Threshold T={T_threshold:.3f}: MSE = {current_mse:.6f}, PSNR = {current_psnr:.2f}dB")

    # Plot MSE & PSNR vs. Threshold
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel("Threshold Value T for |H| in Inverse Filter")
    ax1.set_ylabel("Mean Squared Error (MSE)", color=color)
    ax1.plot(threshold_values_inv, mse_vs_threshold, marker='o', color=color, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel("PSNR (dB)", color=color)
    ax2.plot(threshold_values_inv, psnr_vs_threshold, marker='x', color=color, label='PSNR')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.suptitle("MSE & PSNR of Restored Image vs. Inverse Filter Threshold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Display selected restored images
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1); plt.imshow(img_ny_float, cmap='gray', vmin=0, vmax=1); plt.title('Original Clean'); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(blurred_img_b_clipped, cmap='gray', vmin=0, vmax=1); plt.title(f'Blurred (PSNR: {psnr_blurred:.2f}dB)'); plt.axis('off')
    
    if 0.0 in restored_image_cache: # "No threshold" case
        img_no_thresh = restored_image_cache[0.0]
        psnr_no_thresh = psnr(img_ny_float, img_no_thresh)
        plt.subplot(1, 4, 3); plt.imshow(img_no_thresh, cmap='gray', vmin=0, vmax=1); plt.title(f'Restored (No Threshold)\nPSNR: {psnr_no_thresh:.2f}dB'); plt.axis('off')

    # Find and display best thresholded result (excluding T=0 if other T values exist)
    best_T_val_for_display = -1
    if len(threshold_values_inv) > 1: # If T=0 and other thresholds were tested
        non_zero_T_psnrs = psnr_vs_threshold[1:]
        best_psnr_idx = np.argmax(non_zero_T_psnrs)
        best_T_val_for_display = threshold_values_inv[best_psnr_idx + 1]
    elif len(threshold_values_inv) == 1 and threshold_values_inv[0] != 0.0: # Only one non-zero T
        best_T_val_for_display = threshold_values_inv[0]

    if best_T_val_for_display != -1 and best_T_val_for_display in restored_image_cache:
        img_best_T = restored_image_cache[best_T_val_for_display]
        psnr_best_T = psnr(img_ny_float, img_best_T)
        plt.subplot(1, 4, 4); plt.imshow(img_best_T, cmap='gray', vmin=0, vmax=1); plt.title(f'Restored (Best T={best_T_val_for_display:.3f})\nPSNR: {psnr_best_T:.2f}dB'); plt.axis('off')
    elif len(threshold_values_inv) == 1 and threshold_values_inv[0] == 0.0 and 0.0 in restored_image_cache: # Only T=0 was shown in 3rd plot
         plt.subplot(1, 4, 4); plt.text(0.5, 0.5, 'N/A for best T>0', ha='center', va='center'); plt.title('Best T>0 N/A'); plt.axis('off')


    plt.suptitle('Part B: Inverse Filtering Restoration Examples')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("\nComments on Part B - Inverse Filtering:")
    print("  - The estimated Point Spread Function (PSF), `h`, is obtained by applying the unknown system `psf_transform_function` to an impulse image. Its frequency response, `H`, typically shows how different frequencies are attenuated by the blur (often a low-pass characteristic).")
    print("  - **No Threshold (T=0):** When inverse filtering is applied without a threshold (or T=0), the operation involves dividing the spectrum of the blurred image `G(u,v)` by `H(u,v)`. If `|H(u,v)|` is very small or zero for certain frequencies, this division results in extreme amplification of those frequency components. Any noise present in the blurred image (even minor sensor noise or quantization errors) gets massively amplified, typically rendering the restored image unusable and dominated by noise. The MSE is usually very high, and PSNR very low.")
    print("  - **With Threshold (T > 0):** Using a threshold `T` means that if `|H(u,v)| < T`, the problematic division `1/H(u,v)` is avoided (e.g., by setting the inverse term to 0 for these frequencies). This prevents the excessive amplification of noise at frequencies where the original signal was heavily attenuated by the blur.")
    print("    - An optimal threshold `T` aims to balance the recovery of blurred details (deblurring) with the suppression of noise. If `T` is too small, noise amplification can still be significant. If `T` is too large, too many valid signal frequencies might be discarded, leading to a loss of detail or an overly smoothed restoration.")
    print("    - The plot of MSE (or PSNR) against different threshold values helps in identifying a suitable `T` that minimizes the restoration error (or maximizes PSNR).")
    print("  - Inverse filtering is inherently sensitive to noise and the accuracy of the estimated `H`. The thresholding technique provides a basic form of regularization to make the process more stable.")

if __name__ == '__main__':
    main()