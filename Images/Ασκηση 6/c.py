import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
try:
    img = cv2.imread('hallway.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Error: hallway.png not found. Please make sure the file is in the correct directory.")
    exit()

# 1. Sobel Edge Detection
# Apply Sobel operator in X and Y directions
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of the gradient
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize to 0-255 range for display
sobel_magnitude_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 2. Global Thresholding
# Justification for threshold value:
# A common starting point for thresholding edge magnitudes is a value around the mean or median of the
# non-zero gradient magnitudes, or a value that visually separates strong edges from noise.
# For this example, we'll try a value that's often a reasonable starting point, like 50-100
# on a 0-255 scale. Let's choose 75. This value might need adjustment based on visual inspection
# to get a good balance between detecting significant edges and suppressing noise.
# Another approach is to use Otsu's thresholding if the histogram of gradient magnitudes is bimodal.
# For simplicity, we'll use a fixed threshold here.
threshold_value = 75
_, sobel_thresholded = cv2.threshold(sobel_magnitude_normalized, threshold_value, 255, cv2.THRESH_BINARY)

# Display results for Sobel and Thresholding
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_magnitude_normalized, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_thresholded, cmap='gray')
plt.title(f'Sobel Thresholded (Thresh={threshold_value})')
plt.axis('off')

plt.suptitle('Sobel Edge Detection and Thresholding')
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()


# Bonus: Hough Transform for Line Detection
# Apply Hough Line Transform on the thresholded Sobel image
# Parameters for HoughLinesP:
# rho: Distance resolution of the accumulator in pixels.
# theta: Angle resolution of the accumulator in radians.
# threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes.
# minLineLength: Minimum line length. Line segments shorter than this are rejected.
# maxLineGap: Maximum allowed gap between points on the same line to link them.

# Convert original image to color to draw lines in color
img_color_hough = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_color_hough_copy = img_color_hough.copy() # Work on a copy

# Use the Canny edge detector output for Hough Transform as it's generally preferred
# Canny provides thin, one-pixel wide edges.
edges_canny = cv2.Canny(img, 50, 150, apertureSize=3) # Using Canny edges for Hough

lines = cv2.HoughLinesP(edges_canny, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_color_hough_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw lines in red

# Display Hough Transform results
plt.figure(figsize=(10, 7))

plt.subplot(1, 2, 1)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny Edges for Hough Transform')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_color_hough_copy, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
plt.title('Detected Lines with Hough Transform')
plt.axis('off')

plt.suptitle('Hough Transform Line Detection')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("Edge detection and Hough transform script finished.")
print("Comments on thresholding:")
print(f"- The threshold value for Sobel ({threshold_value}) was chosen empirically. A lower threshold might detect more (weaker) edges, while a higher one will detect only stronger edges.")
print("- For Hough Transform, Canny edges are often preferred over Sobel directly because Canny produces cleaner, thinner edges which can lead to more precise line detection.")
print("- The parameters for HoughLinesP (rho, theta, threshold, minLineLength, maxLineGap) significantly affect the outcome and often require tuning for a specific image or application.")
