import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    """
    CNN Architecture based on the provided diagram:
    Input: 28x28 grayscale images
    
    Architecture:
    1. Conv2d(1, 6, 3x3) + ReLU + AvgPool2d(2x2, stride=2)
    2. Conv2d(6, 16, 3x3) + ReLU + AvgPool2d(2x2, stride=2) 
    3. Flatten + FC(120) + ReLU
    4. FC(84) + ReLU
    5. FC(10) + SoftMax
    """
    
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
    
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        # First convolutional block
        x = self.conv1(x)  # (batch, 6, 26, 26)
        x = F.relu(x)
        x = self.pool1(x)  # (batch, 6, 13, 13)
        
        # Second convolutional block
        x = self.conv2(x)  # (batch, 16, 11, 11)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, 16, 5, 5)
        
        # Flatten for fully connected layers
        x = x.view(-1, 16 * 5 * 5)  # (batch, 400)
        
        # First fully connected layer
        x = self.fc1(x)  # (batch, 120)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second fully connected layer
        x = self.fc2(x)  # (batch, 84)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation, will be applied in loss function)
        x = self.fc3(x)  # (batch, 10)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing feature maps from different layers
        """
        feature_maps = {}
        
        # First conv layer
        x = self.conv1(x)
        feature_maps['conv1'] = x.clone()
        x = F.relu(x)
        feature_maps['conv1_relu'] = x.clone()
        x = self.pool1(x)
        feature_maps['pool1'] = x.clone()
        
        # Second conv layer
        x = self.conv2(x)
        feature_maps['conv2'] = x.clone()
        x = F.relu(x)
        feature_maps['conv2_relu'] = x.clone()
        x = self.pool2(x)
        feature_maps['pool2'] = x.clone()
        
        return feature_maps

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model):
    """
    Print detailed information about the model architecture
    
    Args:
        model: PyTorch model
    """
    print("="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    print(model)
    print("="*50)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("="*50)
    
    # Print layer-wise parameter count
    print("\nLayer-wise parameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")

if __name__ == "__main__":
    # Test the model
    model = MNISTConvNet()
    print_model_info(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature map extraction
    feature_maps = model.get_feature_maps(dummy_input)
    print(f"\nFeature map shapes:")
    for name, fm in feature_maps.items():
        print(f"{name}: {fm.shape}")