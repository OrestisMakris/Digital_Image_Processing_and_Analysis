import sys
import os


#
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from CNN.data_loader import load_mnist_data
from hog_features import visualize_hog_comparison, extract_hog_features_custom
from svm_classifier import train_svm, predict_svm
from evaluation_hog import compute_confusion_matrix, plot_confusion_matrix, print_report

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Φόρτωμα MNIST
    train_loader, test_loader, classes = load_mnist_data(batch_size=256)

    import random
    imgs, _ = next(iter(train_loader))
    arr = imgs[random.randrange(len(imgs))].squeeze().numpy()
    arr = (arr * 0.3081 + 0.1307)  # denormalize
    img_u8 = (arr * 255).astype(np.uint8)
    visualize_hog_comparison(img_u8, patch_size=8)


    X_train, y_train = extract_hog_features_custom(train_loader, patch_size=8)
    X_test,  y_test  = extract_hog_features_custom(test_loader,  patch_size=8)

    
    svm = train_svm(X_train, y_train, C=1.0, kernel='linear')


    y_pred = predict_svm(svm, X_test)
    print_report(y_test, y_pred, classes)

    cm = compute_confusion_matrix(y_test, y_pred, classes, normalize=False)
    plot_confusion_matrix(cm, classes, title='Confusion Matrix', fmt='d')

    cmn = compute_confusion_matrix(y_test, y_pred, classes, normalize=True)
    plot_confusion_matrix(cmn, classes, title='Normalized CM', fmt='.2f')

if __name__ == "__main__":
    
    main()