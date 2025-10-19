import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import numpy as np
from tqdm import tqdm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, batch_norm=True, dropout=0.):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(activation())

        if dropout > 0.:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(784, 200)
        self.layer2 = torch.nn.Linear(200, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.sigmoid(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1, 200)
        self.layer2 = torch.nn.Linear(200, 784)

    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.layer1(x))
        x = torch.nn.functional.sigmoid(self.layer2(x))
        # x = torch.nn.functional.sigmoid(x)
        return x

if __name__ == "__main__":
    device = 'cpu'

    if torch.mps.is_available():
        device = 'mps'

    if torch.cuda.is_available():
        device = 'cuda'

