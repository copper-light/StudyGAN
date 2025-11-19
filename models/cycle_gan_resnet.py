import itertools
import torch
import torch.nn as nn

from functools import partial

from models.cycle_gan_unet import CycleGAN, Downsample, Upsample, ConvBlock

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding='same', padding_mode='zeros'):
        super(ResidualBlock, self).__init__()
        layers = []
        layers += [ConvBlock(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, norm='instance', activation='relu')]
        layers += [ConvBlock(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, norm='instance', activation=None)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, n_filters):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, n_filters, kernel_size=7, stride=1, padding_mode='reflect'),
            Downsample(n_filters, n_filters * 2),
            Downsample(n_filters * 2, n_filters * 4),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            Upsample(n_filters * 4, n_filters * 2),
            Upsample(n_filters * 2, n_filters),
            ConvBlock(n_filters, 3, kernel_size=7, stride=1, activation='tanh', padding_mode='reflect'),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_filters=32):
        super().__init__()

        self.block1 = Downsample(3, n_filters, norm = 'instance', activation = 'lrelu') # 64
        self.block2 = Downsample(n_filters, n_filters * 2, norm = 'instance', activation = 'lrelu') # 32
        self.block3 = Downsample(n_filters * 2, n_filters * 4, norm = 'instance', activation = 'lrelu') # 16
        self.block4 = Downsample(n_filters * 4, n_filters * 8, norm = 'instance', activation = 'lrelu')
        self.block5 = ConvBlock(n_filters * 8,1, kernel_size=4, stride=1, padding='same', norm=None, activation=None)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class CycleGANResNet(CycleGAN):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(CycleGANResNet, self).__init__(input_dim, output_dim, **kwargs)

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G_ab = Generator(self.gen_n_filters).to(device)
        self.D_ab = Discriminator(self.disc_n_filters).to(device)

        self.G_ba = Generator(self.gen_n_filters).to(device)
        self.D_ba = Discriminator(self.disc_n_filters).to(device)

        if is_train:
            adam = partial(torch.optim.Adam, lr=lr, betas=(0.5, 0.999))
            self.G_optimizer = adam(itertools.chain(self.G_ab.parameters(), self.G_ba.parameters()))
            self.D_optimizer = adam(itertools.chain(self.D_ab.parameters(), self.D_ba.parameters()))

            self.criterion_mse = nn.MSELoss()
            self.criterion_l1 = nn.L1Loss()


if __name__ == '__main__':
    gen = Generator(32)

    x = torch.rand(1, 3, 256, 256)
    o = gen(x)
    print(o.shape)

    d = Discriminator()
    print(d(x).shape)