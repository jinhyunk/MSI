import numpy as np
import torch 
import torch.nn as nn

class TanhLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh())
        
    def forward(self, x):
        out = self.NN(x)
        
        return out
    
class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.ReLU())
        
    def forward(self, x):
        out = self.NN(x)
        
        return out