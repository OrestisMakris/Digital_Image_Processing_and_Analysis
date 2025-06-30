import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    """
    Carchitecture based on the provided diagram
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
            x: Input tensor of shape (batch_size 1 28 28)
            
        Returns:
            Output tensor of shape (batch_size 10)
        """

        x = self.conv1(x)  
        x = F.relu(x)
        x = self.pool1(x) 
        x = self.conv2(x)  
        x = F.relu(x)
        x = self.pool2(x)  
        x = x.view(-1, 16 * 5 * 5) 
        x = self.fc1(x)  
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x) 

        
        return x
    
