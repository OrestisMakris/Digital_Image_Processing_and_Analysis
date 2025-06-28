import numpy as np
from skimage.feature import hog

def extract_hog_features(loader, patch_size=8, orientations=9, cells_per_block=(2,2), block_norm='L2-Hys'):
    """
    Εξάγει HOG χαρακτηριστικά από το DataLoader.
    """
    X, y = [], []
    for imgs, labels in loader:
        arrs = imgs.squeeze(1).numpy()            # [B,28,28]
        arrs = arrs * 0.3081 + 0.1307             # denormalize
        for img, lab in zip(arrs, labels.numpy()):
            img_u8 = (img * 255).astype(np.uint8)
            feat = hog(img_u8,
                       orientations=orientations,
                       pixels_per_cell=(patch_size, patch_size),
                       cells_per_block=cells_per_block,
                       block_norm=block_norm)
            X.append(feat); y.append(lab)
    return np.array(X), np.array(y)