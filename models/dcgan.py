import torch
import torch.nn as nn
import torch.optim as optim

from models.valina_gan import GAN


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # 14
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # 7
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1, 1),
            nn.Sigmoid()
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
            nn.LeakyReLU(0.2),

            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 28 -> 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), 64, 7, 7)
        return self.feature(x)

class DCGAN(GAN):

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)

        if is_train:
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = nn.BCELoss()

if __name__ == '__main__':
    gen = Generator()
    seed = torch.randn((64, 100))
    images = gen(seed)
    print(images.shape)

    dis = Discriminator()
    pred = dis(images)
    print(pred.shape)
