"""
@File : EfficientNet.py
@Writer : Wi kyu bok
@Description : EfficientNet 기본 구조에 대한 구현( 적은 파라미터로 image 분류에서 좋은 성능을 달성 )
@Keyword
- 1. width scaling : filter의 수를 늘리는 것
- 2. depth scaling : layer의 수를 늘리는 것
- 3. resolution scaling : input image size를 늘리는 것

- compound scaling : 위의 3가지를 방법을 적절하게 섞어서 활용하는 방법( 이 방법으로 SOTA 달성 )
- resolution scaling이 효과적임(늘리는 비율에 비해 성능개선이 뚜렷함) 
@Date : 22-11-18
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
import math

ap = argparse.ArgumentParser()
ap.add_argument("-mn", "--model_name", required=True, type=str, default="efficientb0",
	help="model name to train")

ap.add_argument("-is", "--image_size", required=True, type=int, default=224,
	help="image size to use to train model \
    efficientb0 : 224 \
    efficientb1 : 240 \
    efficientb2 : 260 \
    efficientb3 : 300 \
    efficientb4 : 380 \
    efficientb5 : 456 \
    efficientb6 : 528 \
    efficientb7 : 600 ")

ap.add_argument("-nc", "--num_classes", required=True, type=int, default=1000,
	help="num classes that model will predict")
args = vars(ap.parse_args())



base_parameter = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3] 
]

phi_values = {
    # alpha, beta, gamma, depth(a**phi)
    "efficientb0" : (0, 224, 0.2),
    "efficientb1" : (0.5, 240, 0.2),
    "efficientb2" : (1, 260, 0.3),
    "efficientb3" : (2, 300, 0.4), 
    "efficientb4" : (3, 380, 0.4),
    "efficientb5" : (4, 456, 0.5),
    "efficientb6" : (5, 528, 0.5),
    "efficientb7" : (5, 600, 0.5)
}

class CNNblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1 ):
        super(CNNblock, self).__init__()
        
        # group=1 로 설정하면 normal conv역할을 수행
        # groups=in_channels로 설정하면 depthwise conv를 수행
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False) # 각 필터, 각 채널에 따라 계산하기 위해 groups를 설정
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # swish랑 동일
        
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C * H * W -> C * 1 * 1 ( 각 채널 별로 하나의 값을 가짐 )
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid() #각 채널값이 0~1값으로
        )
        
    def forward(self, x):
        return x * self.se(x) # 채널에 대한 내부 값의 우선순위를 정함
        

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels/reduction)
    
        if self.expand:
            self.expand_conv = CNNblock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )
            
        self.conv = nn.Sequential(
            CNNblock( hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation( hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def stochastic_depth(self, x):
        if not self.training:
            return x
    
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        
        return torch.div(x, self.survival_prob) *  binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
        
class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = math.ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )
    
    
    def calculate_factors(self,version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNblock(3, channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in base_parameter:
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
            layers_repeats = math.ceil(repeats * depth_factor)
            
            for layer in range(layers_repeats):
                features.append(InvertedResidualBlock(
                    in_channels,
                    out_channels,
                    expand_ratio=expand_ratio,
                    stride=stride if layer == 0 else 1,
                    kernel_size=kernel_size,
                    padding=kernel_size//2, # if k == 1: padding=0, k == 3 : padding=1, k == 5 : padding=2
                        )
                    )
                
                in_channels = out_channels
                
        features.append(
            CNNblock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        
        return nn.Sequential(*features)
    
    def forward(self, x):
        x = self.pool(self.features(x))
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
    


if __name__ == "__main__":
    x = torch.randn(2,3,args['image_size'],args['image_size'])
    model = EfficientNet(args['model_name'], args["num_classes"])
    print(model(x).shape)
    
