"""
@File : ResNet.py
@Writer : Wi kyu bok
@Description : ResNet 기본 구조에 대한 구현
@Date : 22-11-17
"""


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("-mn", "--model_name", required=True, type=str,
	help="model name to train")
ap.add_argument("-ic", "--img_channels", required=True, type=int,
	help="image channels to go through model")
ap.add_argument("-nc", "--num_classes", required=True, type=int,
	help="num classes that model will predict")
args = vars(ap.parse_args())


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_down=None, stride=1):
        super(Resblock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU()
        self.identity_down = identity_down
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_down is not None:
            identity =  self.identity_down(identity)
            
        x += identity
        x = self.relu(x)
        return x
        

class ResNet(nn.Module):
    def __init__(self, res_block, num_layers, img_channels, num_classes):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layer1 = self._create_layer(res_block, num_layers[0], out_channels=64, stride=1)
        self.layer2 = self._create_layer(res_block, num_layers[1], out_channels=128, stride=2)
        self.layer3 = self._create_layer(res_block, num_layers[2], out_channels=256, stride=2)
        self.layer4 = self._create_layer(res_block, num_layers[3], out_channels=512, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.classifier(x)
        
        return x        

    
    def _create_layer(self, block, num_resblock, out_channels, stride):
        identity_down = None
        layers = []
        
        if (stride != 1) or (self.in_channels != out_channels * 4): # 마지막 layer가 아니면
            identity_down = nn.Sequential(
                                nn.Conv2d(in_channels= self.in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride),
                                nn.BatchNorm2d(out_channels*4)
                                    )
        layers.append(block(in_channels=self.in_channels, out_channels=out_channels, identity_down=identity_down, stride=stride))
        self.in_channels = out_channels*4
        
        for _ in range(num_resblock - 1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(Resblock, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(Resblock, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(Resblock, [3,8,36,3], img_channels, num_classes)


if __name__ == "__main__":
    
    
    x = torch.randn(2,3,224,224)
    if args["model_name"] =="resnet50": model = ResNet50(img_channels=args["img_channels"], num_classes=args['num_classes'])
    elif args["model_name"] == "resnet101": model = ResNet101(img_channels=args["img_channels"], num_classes=args['num_classes'])
    else: model = ResNet152(img_channels=args["img_channels"], num_classes=args['num_classes'])
    print(model(x).shape)
