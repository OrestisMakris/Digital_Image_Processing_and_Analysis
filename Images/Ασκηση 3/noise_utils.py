import numpy as np

import matplotlib.pyplot as plt
import scipy.io
import cv2

MAXval = 1.0


def peak_signal_noise_ratio(original, othjer):
    
    """Computes the PSNR between two images using OpenCV, assuming MAXval = 1.0."""

    othjer_ = np.clip(othjer, 0, 1.0).astype(np.float32)

    psnr= cv2.PSNR(original, othjer_, R=MAXval) 
    
    return psnr

def add_gaussian_noise(image, snr_db):

    """Adds Gaussian noise to an image to achieve a specified SNR."""

    signal_power = np.mean(image ** 2)
    noise_power = (signal_power / (10 ** (snr_db / 10)))
    noise_std = np.sqrt(noise_power)
    imageSize = image.shape

    noise = np.random.normal(0, noise_std, imageSize)

    #vazoume ton thoribo mas
    noisy_image = image + noise

    return noisy_image

def add_saltpepper_noise(image, amount_noizz):


    noisy_image_saltpepper_noise = image.copy()

    imageSize = image.size

    num_pixels_to_affect = int(amount_noizz * imageSize)


    num_salt = num_pixels_to_affect // 2

    salt_r = np.random.randint(0, image.shape[0], num_salt)
    salt_c = np.random.randint(0, image.shape[1], num_salt)

    noisy_image_saltpepper_noise[salt_r, salt_c] = MAXval 

    num_pepper = num_pixels_to_affect - num_salt 

    pepper_r = np.random.randint(0, image.shape[0], num_pepper)
    pepper_c = np.random.randint(0, image.shape[1], num_pepper)

    noisy_image_saltpepper_noise[pepper_r, pepper_c] = 0
    
    return noisy_image_saltpepper_noise


def moving_average_filter(image_noise, kernel_size=3):

    fimage = cv2.blur(image_noise, (kernel_size, kernel_size))
    return fimage

def median_filter(image_noise, kernel_size=3):


    image_noise_float32 = image_noise.astype(np.float32)

    filtered_image = cv2.medianBlur(image_noise_float32, kernel_size)
    return filtered_image
