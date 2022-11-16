import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CFG():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 784
    num_classes = 10
    lr = 5e-4
    batch_size = 64
    num_epochs = 1

class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        # n_out = ((n_in + 2p - k) / s) + 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=0) # [b, 6, 28, 28]
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding=0) # [b, 16, 14, 14]
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=1, padding=0) # [b, 120, 10, 10]
        self.avgpool = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2), padding=0) # 14
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1) # b * 120 * 1 * 1 -> b * 120
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    
x = torch.randn(64, 1, 32, 32)
x1 = torch.randn(64, 1, 32, 32)
lenet_model = LeNet(1,10)
print(lenet_model)
x = lenet_model.conv1(x)
print(f"conv1 output shape: {x.shape}")
x = lenet_model.avgpool(lenet_model.relu(x))
print(f"conv1 avg pool output shape: {x.shape}")
x = lenet_model.conv2(x)
print(f"conv2 output shape: {x.shape}")
x = lenet_model.avgpool(lenet_model.relu(x))
print(f"conv2 avg pool output shape: {x.shape}")
x = lenet_model.conv3(x)
print(f"conv3 output shape: {x.shape}")
x = x.reshape(x.shape[0],-1)
print(f"x.reshape: {x.shape}")
x = lenet_model.relu(lenet_model.fc1(x))
print(f"fc1 output shape: {x.shape}")
x = lenet_model.fc2(x)
print(f"fc2 output shape: {x.shape}")
final = lenet_model(x1) # [64, 10]
print(f"lenet final output shape: {final.shape}")
        