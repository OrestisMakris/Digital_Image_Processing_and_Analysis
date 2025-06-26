import numpy as np
from hog_ext import compute_hog
from torchvision import datasets, transforms

def extract_hog_dataset(train=True):
    ds = datasets.MNIST(root='./data', train=train, download=True)
    images = ds.data.numpy()
    labels = ds.targets.numpy()
    feats = [compute_hog(img) for img in images]
    return np.vstack(feats), labels