"""
@File : bert.py
@Writer : Wi kyu bok
@Description : transformer model인 bert에 대해 간단한 구현 및 이해 
@Date : 22-11-19
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

ap.add_argument("-nc", "--num_classes", required=True, type=int, default=1000,
	help="num classes that model will predict")
args = vars(ap.parse_args())


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # 256 embed_size // 8
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divide by heads" 
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc = nn.Linear(self.heads*self.head_dim, self.embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # 
        
        # split embedding into self.heads 
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        