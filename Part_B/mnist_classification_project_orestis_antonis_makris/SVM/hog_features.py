import numpy as np
import matplotlib.pyplot as plt
import os


def extract_hog_features_custom(loader, patch_size=8, orientations=9):

    features = []  
    labels = []     
    for imgs, lbls in loader:
        for img, lbl in zip(imgs, lbls):
            arr = img.squeeze().numpy()
            arr = (arr * 0.3081 + 0.1307)
            img_u8 = (arr * 255).astype(np.uint8)
            feat, _ = compute_hog_scratch(img_u8, patch_size=patch_size, orientations=orientations)
            print("HOG feature shape:", feat.shape)
            features.append(feat)
            labels.append(lbl.item())
    return np.array(features), np.array(labels)


def compute_hog_scratch(img, patch_size=8, orientations=9, block_size=2):
    """
    Υπολογίζει HO χαρακτηριστικά από το μδέν για μία εικόνα 2d np.uint8.
    Επιστρέφι το HO διάνυσμα για το hof
    """
    
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 180  # unsigned


    h, w = img.shape
    n_cells_x = w // patch_size
    n_cells_y = h // patch_size
    cell_hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_edges = np.linspace(0, 180, orientations+1)

    for i in range(n_cells_y):

        for j in range(n_cells_x):

            cell_mag = magnitude[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            cell_ori = orientation[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

            hist = np.zeros(orientations, dtype=np.float32)

            for m, o in zip(cell_mag.flatten(), cell_ori.flatten()):



                bin_idx = int(o // (180/orientations))
                next_bin = (bin_idx + 1) % orientations
                ratio = (o - bin_edges[bin_idx]) / (180/orientations)
                hist[bin_idx] += m * (1 - ratio)
                hist[next_bin] += m * ratio

            cell_hist[i, j, :] = hist


    eps = 1e-6
    hog_vector = []

    for i in range(n_cells_y - block_size + 1):

        for j in range(n_cells_x - block_size + 1):

            block = cell_hist[i:i+block_size, j:j+block_size, :].flatten()
            block = block / (np.linalg.norm(block) + eps)
            hog_vector.extend(block)

    # 4. Visualization
    hog_image = np.zeros_like(img, dtype=np.float32)
    for i in range(n_cells_y):

        for j in range(n_cells_x):

            center_y = i*patch_size + patch_size//2
            center_x = j*patch_size + patch_size//2

            for o in range(orientations):

                angle = bin_edges[o] + (180/orientations)/2
                angle_rad = np.deg2rad(angle)
                
                length = cell_hist[i, j, o] / (cell_hist[i, j, :].max() + eps) * (patch_size//2)


                y1 = int(center_y - length * np.sin(angle_rad))
                x1 = int(center_x - length * np.cos(angle_rad))
                y2 = int(center_y + length * np.sin(angle_rad))
                x2 = int(center_x + length * np.cos(angle_rad))



                if 0 <= y1 < h and 0 <= x1 < w and 0 <= y2 < h and 0 <= x2 < w:

                    rr, cc = np.linspace(y1, y2, patch_size), np.linspace(x1, x2, patch_size)
                    hog_image[rr.astype(int), cc.astype(int)] += 1.0

    return np.array(hog_vector), hog_image

def visualize_hog_comparison(img, patch_size=8, orientations=9, filename=None):


    feat, hog_img = compute_hog_scratch(img, patch_size, orientations)
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(hog_img, cmap='inferno')
    axes[1].set_title(f'HOG (scratch)')
    axes[1].axis('off')
    plt.tight_layout()

    if filename is None:

        filename = "plots/hog_comparison_scratch.png"


        
    plt.savefig(filename)
    plt.close()
    
    return feat

