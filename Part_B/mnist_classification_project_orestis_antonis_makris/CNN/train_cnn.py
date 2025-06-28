import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data_loader import load_mnist_data
from cnn_model import MNISTConvNet

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3, device='cuda'):
    """Runs the full training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"Starting training on {device} for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
    
    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    print(f"Best test accuracy: {max(history['test_acc']):.2f}%")
    return history

def plot_training_history(history):
    """
    Plots training and test loss and accuracy
    with a dark cyan / dark red theme.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Losses
    ax1.plot(epochs, history['train_loss'],
             color='darkcyan', linestyle='-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['test_loss'],
             color='darkred', linestyle='--', linewidth=2, label='Test Loss')
    ax1.set_title('Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracies
    ax2.plot(epochs, history['train_acc'],
             color='darkcyan', linestyle='-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, history['test_acc'],
             color='darkred', linestyle='--', linewidth=2, label='Test Acc')
    ax2.set_title('Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def save_model(model, filepath):
    """Saves only the model's weights."""
    torch.save(model.state_dict(), filepath)
    print(f"Model state saved to {filepath}")

