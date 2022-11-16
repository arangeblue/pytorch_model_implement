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
    input_size = 224
    num_classes = 1000
    lr = 5e-4
    batch_size = 64
    num_epochs = 1
    
    
class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bnorm(self.conv(x)))
    
class Inception_block(nn.Module):
    def __init__(self, in_channels, out_11, red_33, out_33, red_55, out_55, out_11pool):
        super(Inception_block, self).__init__()
        self.first_branch = Conv_block(in_channels=in_channels, out_channels=out_11, kernel_size=(1,1))
        self.second_branch = nn.Sequential(
            Conv_block(in_channels=in_channels, out_channels=red_33, kernel_size=(1,1)),
            Conv_block(in_channels=red_33, out_channels=out_33, kernel_size=(3,3), padding=(1,1))
        )
        
        self.third_branch = nn.Sequential(
            Conv_block(in_channels=in_channels, out_channels=red_55, kernel_size=(1,1)),
            Conv_block(in_channels=red_55, out_channels=out_55, kernel_size=(5,5), padding=(2,2))
        )
        
        self.fourth_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            Conv_block(in_channels=in_channels, out_channels=out_11pool, kernel_size=(1,1))
        )
        
    def forward(self, x):
        return torch.cat([self.first_branch(x), self.second_branch(x), self.third_branch(x), self.fourth_branch(x)], dim=1) # N * filter * 28 * 28

class GoogleNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GoogleNet, self).__init__()
        # n_out = ((n_in + 2p - k) / s) + 1
        self.start_sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3)), # [b,64,112,112]
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2), padding=(1,1)), # [b, 64, 56, 56]
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # [b, 192, 56, 56]
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1)) # [b, 192, 28, 28]
        )

        self.incep3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.incep3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        self.incep4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.incep4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.incep4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.incep4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.incep4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        
        self.incep5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.incep5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.start_sequence(x)
        
        x = self.incep3a(x)        
        x = self.incep3b(x)        
        x = self.maxpool(x)   
        
        x = self.incep4a(x)
        x = self.incep4b(x)
        x = self.incep4c(x)
        x = self.incep4d(x)
        x = self.incep4e(x)
        x = self.maxpool(x)
        
        x = self.incep5a(x)
        x = self.incep5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    

if __name__ == "__main__":
    
    img = torch.randn(2,3,224,224)
    googlenet = GoogleNet(in_channels=3, num_classes=1000)
    print(googlenet(img).shape)
        
        