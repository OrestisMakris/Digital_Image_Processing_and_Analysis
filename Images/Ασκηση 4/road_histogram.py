import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(ax, image, title):

    """Calculates and plots the histogram of a grayscale image."""
   

    image = image.astype(np.uint8)
        
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax.plot(hist, color='darkblue')
    ax.set_title(title)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("frrequency")
    ax.set_xlim([0, 200])

def process_image(image_path):
    """Loads an image, performs histogram analysis and equalization."""
    img_color = cv2.imread(image_path)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)



    fig1, axs1 = plt.subplots(1, 2, figsize=(13, 6))
    axs1[0].imshow(img_gray, cmap='gray')
    axs1[0].set_title('Original Grayscale Image')
    axs1[0].axis('off')
    plot_histogram(axs1[1], img_gray, 'Original Histogram')
    fig1.suptitle(f'Original: {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



    img_global_eq = cv2.equalizeHist(img_gray)

    fig2, axs2 = plt.subplots(1, 2,  figsize=(13, 6))
    axs2[0].imshow(img_global_eq, cmap='gray')
    axs2[0].set_title('Global Histogram Equalization')
    axs2[0].axis('off')
    plot_histogram(axs2[1], img_global_eq, 'Histogram after Global EQ')
    fig2.suptitle(f'Global Equalization: {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



    clahe_clip_limit = 9.0
    clahe_tile_grid_size = (8, 8) 

    print(f"\nApplying Local Histogram Equalization (CLAHE) with tileGridSize={clahe_tile_grid_size} and clipLimit={clahe_clip_limit}")

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    img_local_eq = clahe.apply(img_gray)

    fig3, axs3 = plt.subplots(1, 2 ,  figsize=(13, 6))
    axs3[0].imshow(img_local_eq, cmap='gray')
    axs3[0].set_title(f'Local EQ (CLAHE - Tile: {clahe_tile_grid_size})')
    axs3[0].axis('off')
    plot_histogram(axs3[1], img_local_eq, f'Histogram after Local EQ (CLAHE)')
    fig3.suptitle(f'Local Equalization (CLAHE): {image_path}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



#ορισμός των αρχείων εικόνων

image_files = ['dark_road_1.jpg'    ,  "dark_road_2.jpg" , 'dark_road_3.jpg']


for image_file_path in image_files:
    process_image(image_file_path)
