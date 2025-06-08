import cv2
import numpy as np

import matplotlib.pyplot as plt

#diavazo την εικόνα 
img = cv2.imread("hallway.png",  cv2.IMREAD_GRAYSCALE)
  

# σομπεl 

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3 )
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3 )

# Calculate the magnitude of the gradient
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_magnitude_normalized  = cv2.normalize(sobel_magnitude,  None,  0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


threshold_value =  37

_,  sobel_thresholded = cv2.threshold(sobel_magnitude_normalized, threshold_value, 255, cv2.THRESH_BINARY)


# plot results for Sobel and Thresholding
plt.figure(figsize=(17, 8))

plt.subplot(1,  3, 1)
plt.imshow(img,  cmap="gray")
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_magnitude_normalized,  cmap='gray')
plt.title('Sobel΄λ Edge Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_thresholded, cmap='gray')
plt.title(f'Sobelλ Thresholded (Thresh={threshold_value})')
plt.axis('off')

plt.suptitle('Sobelλ Edge Detection and Thresholding')
plt.tight_layout(rect=[0, 0, 1, 0.9]) 
plt.show()


# το bonus: Hough Transform for Linne Detection


img_color_hough = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color_hough_copy = img_color_hough.copy() # Work on a copy


edges_canny = cv2.Canny(img, 50, 150, apertureSize=3) 

lines = cv2.HoughLinesP(edges_canny, rho=1, theta=np.pi/180, threshold =50, minLineLength=50, maxLineGap=10)

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line[0]

        cv2.line(img_color_hough_copy, (x1, y1),  (x2, y2), (139, 139, 0), 6)

plt.figure(figsize=(10, 7))

# προβολή των αποτελεσμάτων
plt.subplot(1, 2, 1)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny Edges for Hough Transform')

plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_color_hough_copy, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines with Hoough transform')

plt.axis("off")

plt.suptitle('Hough Transform Line detection')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


