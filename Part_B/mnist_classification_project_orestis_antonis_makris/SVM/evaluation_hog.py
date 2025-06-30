import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Ensure the 'plots' directory exists
os.makedirs("plots", exist_ok=True)

def compute_confusion_matrix(y_true, y_pred, classes, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:,None]
    return cm

def plot_confusion_matrix(cm, classes, title='CM', fmt='d', filename=None):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    # Save the plot
    if filename is None:
        filename = f"plots/{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()

def print_report(y_true, y_pred, classes):
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))