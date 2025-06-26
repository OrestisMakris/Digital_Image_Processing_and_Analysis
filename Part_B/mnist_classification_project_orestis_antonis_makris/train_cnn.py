import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

from data_loader import load_mnist_data
from cnn_model import MNISTConvNet

class CNNTrainer:
    """
    Class for training the CNN model using SGD
    """
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.learning_rates = []
        
    def train_epoch(self, optimizer, epoch):
        """
        Train the model for one epoch
        
        Args:
            optimizer: SGD optimizer
            epoch: Current epoch number
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(pbar):


            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            
            # Forward pas
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pas
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """
        Evaluate the model on test set
        
        Returns:
            Test loss and accuracy
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, num_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=1e-4):
        """
        Train the model using SGD
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization weight
        """
      
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
    
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        print(f"Training on device: {self.device}")
        print(f"Training parameters:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Momentum: {momentum}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Batch size: {self.train_loader.batch_size}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(optimizer, epoch)
            
            # Evaluate on test set
            test_loss, test_acc = self.evaluate()
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            self.learning_rates.append(current_lr)
            
            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}')
            print('-' * 50)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best test accuracy: {max(self.test_accuracies):.2f}%")
        
    def plot_training_history(self):
        """
        Plot training and validation loss and accuracy
        """
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
        ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot learning rate
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot loss difference
        loss_diff = [abs(train - test) for train, test in zip(self.train_losses, self.test_losses)]
        ax4.plot(epochs, loss_diff, 'm-', linewidth=2)
        ax4.set_title('Train-Test Loss Difference', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Loss - Test Loss|')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'learning_rates': self.learning_rates
        }, filepath)
        print(f"Model saved to {filepath}")

def main():
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader, classes = load_mnist_data(batch_size=64)
    
    # Create model
    print("Creating CNN model...")
    model = MNISTConvNet()
    
    # Create trainer
    trainer = CNNTrainer(model, train_loader, test_loader, device)
    
    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=15,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Plot results
    trainer.plot_training_history()
    
    # Save model
    trainer.save_model('mnist_cnn_model.pth')

if __name__ == "__main__":
    main()