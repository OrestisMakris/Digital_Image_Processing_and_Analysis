import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def load_mnist_data(batch_size=64, download=True):
    """
    Load MNIT dataset and create data loaders.
    
    Args:
        batch_size: Size of mini-batches.
        download : Whether to download MNIST if not present.
    
    Returns:
        train_loader, test_loader, classes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=download, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = [str(i) for i in range(10)]
    
    return train_loader, test_loader, classes

def visualize_mnist_samples(train_loader, classes):
    """
    Visualize oe sample from each MNIST class.
    
    Args:
        train_loader: Training dat loader.
        classes: List of class names.
    """
    images, labels = next(iter(train_loader))
    
    class_examples = {}
    for i, label in enumerate(labels):
        label_item = label.item()
        if label_item not in class_examples and len(class_examples) < 10:
            class_examples[label_item] = images[i]
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MNIST Dataset - One Sample per Class', fontsize=16)
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        if i in class_examples:
            img = class_examples[i].squeeze()
            img = img * 0.3081 + 0.1307
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Class: {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_batch_samples(train_loader, num_samples=16):
    """
    Visualize a batch of MNIST samples.
    
    Args:
        train_loader: Training data loader.
        num_samples: Number of samples to show
    """
    images, labels = next(iter(train_loader))
    
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle('MNIST Training Samples', fontsize=16)
    
    for i in range(num_samples):
        ax = axes[i // grid_size, i % grid_size]
        img = images[i].squeeze()
        img = img * 0.3081 + 0.1307
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    
    for i in range(num_samples, grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

