import itertools
import torch
import torch.nn as nn

from functools import partial

from models.cycle_gan_unet import CycleGAN, Discriminator, Downsample, Upsample

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding='same', padding_mode='zeros'):
        super(ResidualBlock, self).__init__()
        block = nn.Sequential()
        block.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode))
        block.append(nn.InstanceNorm2d(output_channels))
        block.append(nn.ReLU())
        block.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding='same', padding_mode=padding_mode))
        block.append(nn.InstanceNorm2d(output_channels))
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, n_filters):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            Downsample(3, n_filters, kernel_size=7, stride=1, padding_mode='reflect'),
            Downsample(n_filters, n_filters * 2, kernel_size=3, stride=2),
            Downsample(n_filters * 2, n_filters * 4, kernel_size=3, stride=2),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect'),
            Upsample(n_filters * 4, n_filters * 2, kernel_size=3),
            Upsample(n_filters * 2, n_filters, kernel_size=3),
            Downsample(n_filters, 3, kernel_size=7, stride=1, activation='tanh', padding_mode='reflect'),
        )

    def forward(self, x):
        return self.model(x)


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