import torch
import torch.nn as nn

from models.valina_gan import GAN

class Downsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=2, padding='same'),
            nn.InstanceNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)

class Upsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=4, dropout_rate = 0):
        super().__init__()
        self.module = nn.Sequential(
            nn.Upsample(2, mode='nearest'),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding='same'),
            nn.InstanceNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        if dropout_rate > 0:
            self.module.append(nn.Dropout(dropout_rate))

    def forward(self, x, skip_x):
        x = self.module(x)
        o = torch.concat([x, skip_x], dim=1)
        return o



class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes = 0):
        super().__init__()

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, output_dim, num_classes = 0):
        super().__init__()

    def forward(self, x):
        return x


class CycleGAN(GAN):
    def __init__(self, input_dim, output_dim, name, num_classes, device, is_train, lr):
        super().__init__(input_dim, output_dim, name, num_classes, device, is_train, lr)

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator(output_dim, num_classes).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)

        if is_train:
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.9, 0.999))
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.9, 0.999))

    def train_generator(self, x, y):
        pass

    def train_discriminator(self, x, y):
        pass

if __name__ == '__main__':
    G = Generator(3, 3)
    D = Discriminator(3, 3)