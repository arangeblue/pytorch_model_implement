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


class VGG16(nn.Module):
    def __init__(self,in_channels, num_classes):
        super(VGG16, self).__init__()
        # n_out = ((n_in + 2p - k) / s) + 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3,3),stride=(1,1), padding=1) # 224-2-3+1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1), padding=1) 
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1) # 112
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1) # 56
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
        
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1) # 28
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        
        
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1) # 14
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*7*7,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool(x)
        
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool(x)
        
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


ARCH = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

class Soft_VGG16(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Soft_VGG16, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(ARCH) 
        
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x)
        
        return x
    

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        
        for x in arch:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                           ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)                
    
    
# x = torch.randn(1,3,224,224)
# VGG16_model = VGG16(3, 10)
# print(VGG16_model(x).shape)

    
# x = torch.randn(1,3,224,224)
# Soft_VGG16_model = Soft_VGG16(3, 10)
# print(Soft_VGG16_model(x).shape)

print(CFG.device)
print(torch.__version__)