"""Implement a RL NN in torch"""

import torch
import torch.nn as nn
import numpy as np

class RL(nn.Module):
    
    def __init__(self,input_shape,n_actions):
        super(RL,self).__init__()
        self.deep = nn.Sequential(
            nn.Linear(input_shape[0],32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )

    def forward(self,x):
        deep_out = self.deep(x)
        return self.fc(deep_out)


'''
Network idea from https://github.com/bentrevett/pytorch-rl/blob/master/1%20-%20Vanilla%20Policy%20Gradient%20(REINFORCE)%20%5BCartPole%5D.ipynb

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

Using 128 hidden dim and 0.5 dropout

'''
