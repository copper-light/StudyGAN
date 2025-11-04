import torch
import torch.nn as nn
import torch.optim as optim

from models.valina_gan import GAN


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes = 0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = input_dim[0]
        self.w = input_dim[1]
        self.h = input_dim[2]
        self.out_features = self.in_channels * self.w * self.h
        self.in_features = self.out_features + num_classes
        self.input = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.LeakyReLU(0.2),
        )

        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(self.in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
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
            nn.Linear(int((self.w * self.h)/16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), self.in_channels, self.h, self.w)
        x = self.feature(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, output_dim, num_classes = 0):
        super().__init__()
        self.num_classes = num_classes

        self.output_channels = output_dim[0]
        self.w = output_dim[1]
        self.h = output_dim[2]
        self.out_features = int((self.w * self.h) / (4 * 4) * 64)

        self.input = nn.Sequential(
            nn.Linear(100 + num_classes, self.out_features),
            nn.BatchNorm1d(self.out_features),
            nn.ReLU()
        )

        self.feature = nn.Sequential(
            # * 2
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # * 2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # * 1
            nn.Conv2d(64, self.output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.Sigmoid() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), 64, int(self.w/4), int(self.h/4))
        return self.feature(x)


class DCGAN(GAN):

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator(output_dim, num_classes).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)

        if is_train:
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = nn.BCELoss()


if __name__ == '__main__':
    gen = Generator(10)
    seed = torch.randn((64, 110))
    images = gen(seed)
    print(images.shape)

    images = images.reshape(64, (28*28))
    onehot = torch.randn(64, 10)
    images = torch.concat((images, onehot), dim=1)
    dis = Discriminator(10)
    pred = dis(images)
    print(pred.shape)
