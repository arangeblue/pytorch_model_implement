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
    num_epochs = 2
    
    


# Simple Neural Network
class Simple_NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Simple_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
        
# Simple Convolution Neural Network
class Simple_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Simple_CNN, self).__init__()
        """
        1. number of features
        n_out = ((n_in + 2p - k) / s) + 1
        n_in : number of input features
        n_out : number of output features
        k : convolution kernel size
        s : convolution stride size
        p : convolution padding size
        
        example : mnist input size (1, 28, 28)
        -> ((28 * (2*1) - 3) / 1) + 1) : 28 
        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        self.fc1 = nn.Linear(32*7*7, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x) # [B, 16, 28, 28]
        x = self.relu(x) 
        x = self.pool(x) # [B, 16, 14, 14]
        x = self.conv2(x) # [B, 32, 14, 14]
        x = self.relu(x) 
        x = self.pool(x) # [B, 32, 7, 7]
        x = x.reshape(x.shape[0], -1) # flatten # [B, 32*7*7]
        x = self.fc1(x) # [B, num_classes]
        
        return x
    
# x = torch.randn(64, 1, 28, 28)
# NN_model = Simple_NN(784, 10)
# print(NN_model(x.reshape(x.shape[0],-1)).shape) # [64, 10]
    
# CNN_model = Simple_CNN(1,10)
# print(CNN_model(x).shape) # [64, 10]

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=CFG.batch_size, shuffle=False)

NN_model = Simple_NN(input_size=CFG.input_size, num_classes=CFG.num_classes).to(CFG.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NN_model.parameters(), lr=CFG.lr)

# NN_epochs
for epoch in range(CFG.num_epochs):
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(CFG.device)
        target = target.to(CFG.device)
        
        data = data.reshape(data.shape[0],-1)
        
        output = NN_model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        
def check_accuracy(loader, model, model_name="nn"):
    if loader.dataset.train:
        print("Checking accuracy an train data")
    else:
        print("Checking accuracy an valid data")
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(CFG.device)
            y = y.to(CFG.device)
            if model_name=="nn":
                x = x.reshape(x.shape[0],-1)            
            output = model(x)
            _, preds = output.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)
            
        
        if model_name=="nn":
            print(f'Simple Neural Network Accuary : {float(num_correct) / float(num_samples)*100:.4f}')
        else:
            print(f'Convolution Neural Network Accuary : {float(num_correct) / float(num_samples)*100:.4f}')
    model.train()
    
check_accuracy(train_loader, NN_model, "nn")
check_accuracy(valid_loader, NN_model, "nn")
        
        
CNN_model = Simple_CNN(in_channels=1, num_classes=CFG.num_classes).to(CFG.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_model.parameters(), lr=CFG.lr)

# NN_epochs
for epoch in range(CFG.num_epochs):
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(CFG.device)
        target = target.to(CFG.device)
        
        output = CNN_model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        

check_accuracy(train_loader, CNN_model, "cnn")
check_accuracy(valid_loader, CNN_model, "cnn")