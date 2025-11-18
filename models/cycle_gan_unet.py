import torch
import torch.nn as nn
import numpy as np
import itertools

from functools import partial

from models.valina_gan import GAN

class Downsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=5, stride=2, norm = True, activation='relu'):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=kernel_size//2),
        )

        if norm:
            self.module.append(nn.InstanceNorm2d(output_channel))

        if activation == 'relu':
            self.module.append(nn.ReLU(inplace=True))
        elif activation == 'lrelu':
            self.module.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'tanh':
            self.module.append(nn.Tanh())

    def forward(self, x):
        return self.module(x)

class Upsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=5, dropout_rate = 0, norm=True, activation='relu', use_skip_connection=True):
        super().__init__()
        self.use_skip_connection = use_skip_connection
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=1, padding='same'),
        )

        if norm:
            self.module.append(nn.InstanceNorm2d(output_channel))

        if activation == 'relu':
            self.module.append(nn.ReLU(inplace=True))
        elif activation == 'tanh':
            self.module.append(nn.Tanh())

        if dropout_rate > 0:
            self.module.append(nn.Dropout(dropout_rate))


    def forward(self, x, skip_x = None):
        x = self.module(x)
        if self.use_skip_connection and skip_x is not None:
            x = torch.concat([x, skip_x], dim=1)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_filters=32):
        super().__init__()
        self.block1 = Downsample(3, n_filters, norm = False, activation = 'lrelu') # 64
        self.block2 = Downsample(n_filters, n_filters * 2, activation='lrelu') # 32
        self.block3 = Downsample(n_filters * 2, n_filters * 4, activation='lrelu') # 16
        self.block4 = Downsample(n_filters * 4, n_filters * 8, stride=1, activation='lrelu')
        self.block5 = Downsample(n_filters * 8, 1, norm=False, stride=1, activation=None)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_filters=32):
        super().__init__()
        self.d1 = Downsample(3, n_filters) # 32
        self.d2 = Downsample(n_filters, n_filters * 2) # 64
        self.d3 = Downsample(n_filters * 2, n_filters * 4) # 128
        self.d4 = Downsample(n_filters * 4, n_filters * 8) # 256

        self.u1 = Upsample(n_filters * 8, n_filters * 4) # 128 + 128
        self.u2 = Upsample(n_filters * 8, n_filters * 2) # 64 + 64
        self.u3 = Upsample(n_filters * 4, n_filters) # 32 + 32
        self.u4 = Upsample(n_filters * 2, 3, norm = False, activation='tanh', use_skip_connection=False)

    def forward(self, x):
        dx1 = self.d1(x)
        dx2 = self.d2(dx1)
        dx3 = self.d3(dx2)
        dx4 = self.d4(dx3)

        ux1 = self.u1(dx4, dx3)
        ux2 = self.u2(ux1, dx2)
        ux3 = self.u3(ux2, dx1)
        o = self.u4(ux3)

        return o


class CycleGAN(GAN):
    def __init__(self, input_dim, output_dim, name, device, is_train, lr, gen_n_filters=32, disc_n_filters=32, lambda_validation = 1, lambda_reconstruction = 10, lambda_identity = 2):
        self.lambda_validation = lambda_validation
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_identity = lambda_identity
        self.patch_size = input_dim[-1] // (2 ** 3)
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters
        super().__init__(input_dim, output_dim, name, 0, device, is_train, lr)

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


    def _train_gen(self, g, gen_x, d, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        fake_x = gen_x(y).detach()
        one = torch.ones(x.size(0), 1, self.patch_size, self.patch_size).to(self.device)

        g.train()
        fake_y = g(x)
        pred = d(fake_y)
        loss_validation = self.criterion_mse(pred, one)

        reconstruction_y = g(fake_x)
        loss_reconstruction = self.criterion_l1(reconstruction_y, y)

        identity_y = g(y)
        loss_identity = self.criterion_l1(identity_y, y)

        loss = (self.lambda_validation * loss_validation) + (self.lambda_reconstruction * loss_reconstruction) + (self.lambda_identity * loss_identity)

        return loss


    def _train_disc(self, g, d, x):
        x = x.to(self.device)
        fake_x = g(x).detach()
        one = torch.ones(x.size(0), 1, self.patch_size, self.patch_size).to(self.device)
        zero = torch.zeros(x.size(0), 1, self.patch_size, self.patch_size).to(self.device)

        pred_real = d(x)
        loss_real = self.criterion_mse(pred_real, one)

        pred_fake = d(fake_x)
        loss_fake = self.criterion_mse(pred_fake, zero)

        return (loss_real + loss_fake) * 0.5

    def train_generator(self, img_a, img_b):
        loss_ab = self._train_gen(self.G_ab, self.G_ba, self.D_ab, img_a, img_b)
        loss_ba = self._train_gen(self.G_ba, self.G_ab, self.D_ba, img_b, img_a)
        loss = (loss_ab + loss_ba)

        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()

        return loss.item() * 0.5, None

    def train_discriminator(self, img_a, img_b):
        loss_a = self._train_disc(self.G_ab, self.D_ab, img_a)
        loss_b = self._train_disc(self.G_ba, self.D_ba, img_b)
        loss = (loss_a + loss_b) * 0.5

        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()

        return loss.item(), None

    def _generate_seed(self, labels):
        return None

    def generate_image_to_numpy(self, img_a, img_b):
        images = []
        self.G_ab.eval()
        self.G_ba.eval()
        with torch.no_grad():
            img_a = img_a.to(self.device)
            img_b = img_b.to(self.device)

            fake_b = self.G_ab(img_a)
            fake_a = self.G_ba(img_b)

            recon_b = self.G_ab(fake_a)
            recon_a = self.G_ba(fake_b)

            iden_b = self.G_ab(img_b)
            iden_a = self.G_ba(img_a)

            images.append(img_a[0].cpu().numpy())
            images.append(fake_b[0].cpu().numpy())
            images.append(recon_a[0].cpu().numpy())
            images.append(iden_a[0].cpu().numpy())

            images.append(img_b[0].cpu().numpy())
            images.append(fake_a[0].cpu().numpy())
            images.append(recon_b[0].cpu().numpy())
            images.append(iden_b[0].cpu().numpy())

        return np.array(images)

    def get_checkpoint(self):
        model = {'G_ab': self.G_ab.state_dict(),
                 'D_ab': self.D_ab.state_dict(),
                 'G_ba': self.G_ba.state_dict(),
                 'D_ba': self.D_ba.state_dict(),
                 'G_optimizer': self.G_optimizer.state_dict(),
                 'D_optimizer': self.D_optimizer.state_dict()}
        return model

    def load_checkpoint(self, checkpoint):
        self.G_ab.load_state_dict(checkpoint['G_ab'])
        self.D_ab.load_state_dict(checkpoint['D_ab'])
        self.G_ba.load_state_dict(checkpoint['G_ba'])
        self.D_ba.load_state_dict(checkpoint['D_ba'])
        self.G_optimizer = torch.optim.Adam(itertools.chain(self.G_ab.parameters(), self.G_ba.parameters()))
        self.D_optimizer = torch.optim.Adam(itertools.chain(self.D_ab.parameters(), self.D_ba.parameters()))
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])



if __name__ == '__main__':
    images = torch.randn(32, 3, 128, 128)

    encoder = Downsample(3, 1)
    output = encoder(images)
    print(output.size())

    G = Generator()
    output = G(images)
    print(output.shape)

    D = Discriminator()
    output = D(output)
    print(output.shape)

    # a = torch.randn(32, 3, 128, 128)
    # b = torch.randn(32, 3, 128, 128)
    #
    # gan = CycleGAN((3, 128, 128), (3, 128, 128), 'cyclegan', 0, None, True, 1e-4)
    # d_loss = gan.train_discriminator(a, b)
    # g_loss = gan.train_generator(a, b)
    # print(d_loss, g_loss)