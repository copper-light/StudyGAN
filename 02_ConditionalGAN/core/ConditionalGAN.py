import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784 + 10, 200)
        self.activation = nn.LeakyReLU(0.02)
        # self.activation = nn.Sigmoid()
        # self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(200)
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100 + 10, 200)
        self.activation = nn.LeakyReLU(0.02)
        # self.activation = nn.Sigmoid()
        # self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(200)
        self.layer2 = nn.Linear(200, 784)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = nn.functional.sigmoid(self.layer2(x))
        return x


if __name__ == "__main__":
    gen = Generator()
    seed = torch.rand(4, 100)
    one_hot = nn.functional.one_hot(torch.tensor([1,2,3,4]), 10).type(torch.FloatTensor)
    seed = torch.concat([seed, one_hot], dim=1)
    print(seed[0])
    print(seed.shape)
    pred = gen(seed)
    print(pred.shape)
