import torch
import torch.nn as nn

import matplotlib.pyplot as plt

device = 'cpu'
# if torch.mps.is_available():
#     device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'


class RandomWeightedAverage():
    def __call__(self, a, b, batch_size):
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        return (alpha * a) + ((1 - alpha) * b)


class GradientPenaltyLoss(nn.Module):
    def forward(self, pred, interpolated_images):
        gradients = torch.autograd.grad(pred, interpolated_images, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
        l2_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1, keepdim=True))
        penalty = torch.mean(torch.sqrt(1 - l2_norm))
        return penalty


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return -torch.mean(preds * targets)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),

            # 14
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),

            # 7
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(100, 3136),
            nn.BatchNorm1d(3136),
            nn.ReLU()
        )

        self.feature = nn.Sequential(

            # 7 -> 14
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 28 -> 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), 64, 7, 7)
        return self.feature(x)


if __name__ == '__main__':
    gen = Generator()
    seed = torch.randn((64, 100))
    images = gen(seed)
    print(images.shape)

    dis = Discriminator()
    pred = dis(images)
    print(pred.shape)
