import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(ax, image, title):
    """Calculates and plots the histogram of a grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim([0, 256])

def process_image(image_path):
    """Loads an image, performs histogram analysis and equalization."""
    try:
        img_color = cv2.imread(image_path)
        if img_color is None:
            print(f"Error: Could not load image {image_path}")
            return
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error loading or converting image {image_path}: {e}")
        return

    print(f"\nProcessing image: {image_path}")

    # 1. Original Image and Histogram
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))
    axs1[0].imshow(img_gray, cmap='gray')
    axs1[0].set_title('Original Grayscale Image')
    axs1[0].axis('off')
    plot_histogram(axs1[1], img_gray, 'Original Histogram')
    fig1.suptitle(f'Original: {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("Comments on Original Histogram:")
    print("- Observe the distribution of pixel intensities. For dark images, the histogram is typically skewed towards the lower (darker) intensity values.")
    print("- A narrow histogram indicates low contrast, meaning the range of intensities used is small.")

    # 2. Global Histogram Equalization
    img_global_eq = cv2.equalizeHist(img_gray)

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    axs2[0].imshow(img_global_eq, cmap='gray')
    axs2[0].set_title('Global Histogram Equalization')
    axs2[0].axis('off')
    plot_histogram(axs2[1], img_global_eq, 'Histogram after Global EQ')
    fig2.suptitle(f'Global Equalization: {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\nComments on Global Histogram Equalization:")
    print("- Global equalization attempts to spread out the most frequent intensity values, resulting in a flatter histogram over the entire intensity range.")
    print("- This generally increases global contrast, making details more visible.")
    print("- However, it can sometimes over-enhance noise or wash out details in regions that are already well-contrasted or very dark/bright, as it doesn't consider local context.")

    # 3. Local Histogram Equalization (CLAHE)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Parameters:
    # clipLimit: Threshold for contrast limiting. Higher values allow more contrast.
    # tileGridSize: Size of the grid for local equalization. (e.g., (8, 8))
    
    # --- Choosing tileGridSize ---
    # A smaller tileGridSize (e.g., (8,8) or (16,16)) focuses on smaller local regions,
    # which can enhance local details better but might also amplify noise.
    # A larger tileGridSize (e.g., (32,32) or (64,64)) behaves more like global equalization.
    # The choice depends on the image content and desired outcome.
    # For road images, a moderate size like (16,16) or (32,32) might be a good starting point.
    # We will use (16,16) as a generally good starting point.
    clahe_clip_limit = 2.0
    clahe_tile_grid_size = (16, 16) # You can experiment with this
    print(f"\nApplying Local Histogram Equalization (CLAHE) with tileGridSize={clahe_tile_grid_size} and clipLimit={clahe_clip_limit}")

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    img_local_eq = clahe.apply(img_gray)

    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
    axs3[0].imshow(img_local_eq, cmap='gray')
    axs3[0].set_title(f'Local EQ (CLAHE - Tile: {clahe_tile_grid_size})')
    axs3[0].axis('off')
    plot_histogram(axs3[1], img_local_eq, f'Histogram after Local EQ (CLAHE)')
    fig3.suptitle(f'Local Equalization (CLAHE): {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\nComments on Local Histogram Equalization (CLAHE):")
    print(f"- CLAHE applies histogram equalization to local regions (tiles) of the image, rather than globally.")
    print(f"- The `tileGridSize` chosen ({clahe_tile_grid_size}) defines these regions. Smaller tiles can adapt better to local changes but might introduce noise if the `clipLimit` is too high.")
    print(f"- The `clipLimit` ({clahe_clip_limit}) restricts contrast amplification to prevent over-enhancement of noise.")
    print("- CLAHE often provides better results than global equalization for images with varying illumination, enhancing local contrast without significantly amplifying noise.")
    print("- The resulting histogram might not be perfectly flat but will show a better distribution of intensities across different parts of the image compared to the original.")
    print("- For these dark road images, CLAHE should help reveal more details in shadowed areas and potentially in the road markings or signs.")


# List of images to process
image_files = ['dark_road_1.jpg', 'dark_road_2.jpg', 'dark_road_3.jpg']

# Check if images exist (optional, but good practice)
import os
for image_file in image_files:
    if not os.path.exists(image_file):
        print(f"Warning: Image file '{image_file}' not found in the current directory. Please ensure it's there.")
        print(f"Current directory: {os.getcwd()}")
        # Attempt to find it in a common 'Images' or 'Ασκηση 4' subdirectory if not in root
        search_paths = [
            os.path.join("Images", "Ασκηση 4", image_file),
            os.path.join("Ασκηση 4", image_file),
            os.path.join("..", "Images", "Ασκηση 4", image_file), # If script is in Ασκηση 4
            image_file # Original path
        ]
        found = False
        for sp in search_paths:
            if os.path.exists(sp):
                image_files[image_files.index(image_file)] = sp # Update path
                found = True
                break
        if not found:
             print(f"Still could not find {image_file}. Please check the path.")


for image_file in image_files:
    # Check again if the path was updated and exists
    if os.path.exists(image_file):
        process_image(image_file)
    else:
        print(f"Skipping {image_file} as it was not found.")

print("\n--- End of Processing ---")