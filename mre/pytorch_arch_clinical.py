import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tensorboardX import SummaryWriter


class Clinical(nn.Module):
    def __init__(self, in_channels, out_channels, n_hidden=32):
        super(Clinical, self).__init__()

        self.fc_in = nn.Linear(in_channels, n_hidden)
        self.fc_1 = nn.Linear(n_hidden, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, out_channels)

        self.relu = nn.ReLU()

    def forward(self, dummy_in, x):

        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_out(x)

        return x
