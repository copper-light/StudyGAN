import itertools
import torch
import torch.nn as nn

from functools import partial

from models.cycle_gan_unet import CycleGAN, Downsample, Upsample, ConvBlock

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        

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
    def __init__(self, n_filters, n_residual = 7):
        super(Generator, self).__init__()
        layers = []

        layers += [
            ConvBlock(3, n_filters, kernel_size=7, stride=1, norm = 'instance', padding_mode='reflect'),
            Downsample(n_filters, n_filters * 2, norm = 'instance'),
            Downsample(n_filters * 2, n_filters * 4, norm = 'instance')
        ]
        
        for _ in range(n_residual):
            layers += [ResidualBlock(n_filters * 4, n_filters * 4, kernel_size=3, stride=1, padding_mode='reflect')]

        layers += [
            Upsample(n_filters * 4, n_filters * 2, norm = 'instance'),
            Upsample(n_filters * 2, n_filters, norm = 'instance'),
            ConvBlock(n_filters, 3, kernel_size=7, stride=1, activation='tanh', padding_mode='reflect', norm = None)
        ]

        self.model = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_filters=32):
        super().__init__()
        layers = []
        layers += [Downsample(3, n_filters, norm = None, activation = 'lrelu')] # 128
        layers += [Downsample(n_filters, n_filters * 2, norm = 'instance', activation = 'lrelu')] # 56
        layers += [Downsample(n_filters * 2, n_filters * 4, norm = 'instance', activation = 'lrelu')] # 32
        layers += [Downsample(n_filters * 4, n_filters * 8, norm = 'instance', activation = 'lrelu')] # 16 patch
        layers += [ConvBlock(n_filters * 8,1, kernel_size=4, stride=1, padding='same', norm=None, activation=None)]

        self.model = nn.Sequential(*layers)
        self.apply(weights_init_normal)

    def forward(self, x):
        return self.model(x)


class CycleGANResNet(CycleGAN):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(CycleGANResNet, self).__init__(input_dim, output_dim, **kwargs)

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G_ab = Generator(self.gen_n_filters).to(device)
        self.G_ba = Generator(self.gen_n_filters).to(device)

        self.D_a = Discriminator(self.disc_n_filters).to(device)
        self.D_b = Discriminator(self.disc_n_filters).to(device)

        if is_train:
            adam = partial(torch.optim.Adam, lr=lr, betas=(0.5, 0.999))
            self.G_optimizer = adam(itertools.chain(self.G_ab.parameters(), self.G_ba.parameters()))
            self.D_optimizer = adam(itertools.chain(self.D_a.parameters(), self.D_b.parameters()))

            self.criterion_mse = nn.MSELoss()
            self.criterion_l1 = nn.L1Loss()


if __name__ == '__main__':
    gen = Generator(32)

    x = torch.rand(1, 3, 256, 256)
    o = gen(x)
    print(o.shape)

    d = Discriminator()
    print(d(x).shape)