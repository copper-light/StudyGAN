import itertools
import torch
import torch.nn as nn

from functools import partial

from models.cycle_gan_unet import CycleGAN, Discriminator

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        block = nn.Sequential()
        block.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        block.append(nn.InstanceNorm2d(output_channels))
        block.append(nn.ReLU())
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, gen_n_filters):
        super(Generator, self).__init__()
        self.gen_n_filters = gen_n_filters

    def forward(self, x):
        pass


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