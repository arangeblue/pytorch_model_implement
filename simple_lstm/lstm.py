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
    input_size = 28
    sequence_length = 28
    num_layers = 2
    hidden_size = 128
    num_classes = 10
    lr = 5e-4
    batch_size = 64
    num_epochs = 2
    
    
# bidirectional LSTM    
class BILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BILSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x):
        """
        1. h0 : (num_layer * num_directions, batch, hidden_size)
        2. c0 : (num_layer * num_directions, batch, hidden_size)

        Args:
            x (_type_): sequential data

        Returns:
            out: (seq_len, batch, num_direction * hidden_size) -> (batch, classes)
            
        """
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(CFG.device) # hidden states
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(CFG.device) # cell states
        
        out, _ = self.lstm(x, (h0, c0))
        print(out.shape)
        out = self.fc(out[:, -1, :]) # each last hidden states
        
        return out
        

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=CFG.batch_size, shuffle=False)

BILSTM_model = BILSTM(input_size=CFG.input_size,hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, num_classes=CFG.num_classes).to(CFG.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(BILSTM_model.parameters(), lr=CFG.lr)

# BILSTM epochs
for epoch in range(CFG.num_epochs):
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(CFG.device).squeeze(1)
        target = target.to(CFG.device)
        
        
        output = BILSTM_model(data)
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
            elif model_name=="bilstm":
                x = x.squeeze(1)
            output = model(x)
            _, preds = output.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)
            
        
        if model_name=="nn":
            print(f'Simple Neural Network Accuary : {float(num_correct) / float(num_samples)*100:.4f}')
        elif model_name == "cnn":
            print(f'Convolution Neural Network Accuary : {float(num_correct) / float(num_samples)*100:.4f}')
        else:
            print(f'BILSTM Network Accuary : {float(num_correct) / float(num_samples)*100:.4f}')
            
    model.train()
    
print(CFG.device)
check_accuracy(train_loader, BILSTM_model, "bilstm")
check_accuracy(valid_loader, BILSTM_model, "bilstm")