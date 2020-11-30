from typing import List, Union, Dict
import collections
import enum
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision
import pdb
from hparams import hparams
import src.dataset

input_dim = 4
hidden_dim = 32
num_layers = 2
output_dim = 1


class baselineLSTM(nn.Module):
    def __init__(self):
        super(baselineLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.cuda()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc2(out[:, -1, :])
        out = self.fc(out)
        return out


class baselineGRU(nn.Module):
    def __init__(self):
        super(baselineGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.cuda()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc2(out[:, -1, :])
        out = self.fc(out)
        return out


