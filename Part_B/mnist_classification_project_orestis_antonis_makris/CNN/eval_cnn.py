import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

def get_predictions(model, test_loader, device='cuda'):
    """
    Get model predictions for the entire test set.
    """
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(true_labels, predictions, classes, normalize=False, title='Confusion Matrix'):
    """
    Plot confusion matrix using seaborn.
    """
    cm = confusion_matrix(true_labels, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
               xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
def plot_conv_filters(model, num_filters=6):
    """
    Display the first `num_filters` convolutional kernels of conv1.
    """
    # pull out the weight tensor, shape [out_ch, in_ch=1, kH, kW]
    kernels = model.conv1.weight.data.clone().cpu().squeeze(1)
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters*2, 2))
    fig.suptitle('First–Layer Conv Filters', fontsize=16)
    for i in range(num_filters):
        ax = axes[i]
        filt = kernels[i]
        # normalize to [0,1]
        filt = (filt - filt.min()) / (filt.max() - filt.min())
        ax.imshow(filt, cmap='viridis')
        ax.set_title(f'#{i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def print_classification_report(true_labels, predictions, classes):
    """
    Print a detailed classification report from sklearn.
    """
    print("="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(true_labels, predictions, 
                              target_names=classes, digits=4))

def plot_per_class_metrics(true_labels, predictions, classes):
    """
    Plot per-class precision, recall, and F1-score.
    """
    precision = precision_score(true_labels, predictions, average=None)
    recall = recall_score(true_labels, predictions, average=None)
    f1 = f1_score(true_labels, predictions, average=None)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision')
    bars2 = ax.bar(x, recall, width, label='Recall')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()