import torch
import torch.nn as nn
import numpy as np

from models.valina_gan import GAN

class Downsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=5, stride=2, norm = True, activation='relu'):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=2),
        )

        if norm:
            self.module.append(nn.InstanceNorm2d(output_channel))

        if activation == 'relu':
            self.module.append(nn.ReLU(inplace=True))
        elif activation == 'lrelu':
            self.module.append(nn.LeakyReLU(0.2, inplace=True))

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
    def __init__(self):
        super().__init__()
        base_channel = 32
        self.block1 = Downsample(3, base_channel, norm = False, activation = 'lrelu') # 64
        self.block2 = Downsample(base_channel, base_channel * 2, activation='lrelu') # 32
        self.block3 = Downsample(base_channel * 2, base_channel * 4, activation='lrelu') # 16
        self.block4 = Downsample(base_channel * 4, base_channel * 8, stride=1, activation='lrelu')
        self.block5 = Downsample(base_channel * 8, 1, norm=False, stride=1, activation=None)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        base_channel = 32
        self.d1 = Downsample(3, base_channel) # 32
        self.d2 = Downsample(base_channel, base_channel * 2) # 64
        self.d3 = Downsample(base_channel * 2, base_channel * 4) # 128
        self.d4 = Downsample(base_channel * 4, base_channel * 8) # 256

        self.u1 = Upsample(base_channel * 8, base_channel * 4) # 128 + 128
        self.u2 = Upsample(base_channel * 8, base_channel * 2) # 64 + 64
        self.u3 = Upsample(base_channel * 4, base_channel) # 32 + 32
        self.u4 = Upsample(base_channel * 2, 3, norm = False, activation='tanh', use_skip_connection=False)

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
    def __init__(self, input_dim, output_dim, name, device, is_train, lr, w_validation = 1, w_reconstruction = 10, w_identity = 2):
        super().__init__(input_dim, output_dim, name, 0, device, is_train, lr)
        self.w_validation = w_validation
        self.w_reconstruction = w_reconstruction
        self.w_identity = w_identity

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G_ab = Generator().to(device)
        self.D_ab = Discriminator().to(device)

        self.G_ba = Generator().to(device)
        self.D_ba = Discriminator().to(device)

        if is_train:
            self.G_ab_optimizer = torch.optim.Adam(self.G_ab.parameters(), lr=lr, betas=(0.9, 0.999))
            self.D_ab_optimizer = torch.optim.Adam(self.D_ab.parameters(), lr=lr, betas=(0.9, 0.999))

            self.G_ba_optimizer = torch.optim.Adam(self.G_ba.parameters(), lr=lr, betas=(0.9, 0.999))
            self.D_ba_optimizer = torch.optim.Adam(self.D_ba.parameters(), lr=lr, betas=(0.9, 0.999))

            self.criterion_mse = nn.MSELoss()
            self.criterion_l1 = nn.L1Loss()


    def _train_gen(self, g, gen_x, d, opti, x, y):
        fake_x = gen_x(y).detach()
        one = torch.ones(x.size(0), 1, 16, 16).to(self.device)

        g.train()
        fake_y = g(x).detach()
        pred = d(fake_y)
        loss_validation = self.criterion_mse(pred, one)

        reconstruction_y = g(fake_x)
        loss_reconstruction = self.criterion_l1(reconstruction_y, y)

        identity_y = g(y)
        loss_identity = self.criterion_l1(identity_y, y)

        loss = (self.w_validation * loss_validation) + (self.w_reconstruction * loss_reconstruction) + (self.w_identity * loss_identity)

        opti.zero_grad()
        loss.backward()
        opti.step()

        return loss.item()


    def _train_disc(self, g, d, opti, x):
        fake_x = g(x).detach()
        one = torch.ones(x.size(0), 1, 16, 16).to(self.device)
        zero = torch.zeros(x.size(0), 1, 16, 16).to(self.device)

        pred_real = d(x)
        loss_real = self.criterion_mse(pred_real, one)
        opti.zero_grad()
        loss_real.backward()
        opti.step()

        pred_fake = d(fake_x)
        loss_fake = self.criterion_mse(pred_fake, zero)
        opti.zero_grad()
        loss_fake.backward()
        opti.step()

        return (loss_real.item() + loss_fake.item()) * 0.5

    def train_generator(self, img_a, img_b):
        loss_ab = self._train_gen(self.G_ab, self.G_ba, self.D_ab, self.G_ab_optimizer, img_a, img_b)
        loss_ba = self._train_gen(self.G_ba, self.G_ab, self.D_ba, self.G_ba_optimizer, img_b, img_a)
        loss = (loss_ab + loss_ba) * 0.5

        return loss

    def train_discriminator(self, img_a, img_b):
        loss_a = self._train_disc(self.G_ab, self.D_ab, self.D_ab_optimizer, img_a)
        loss_b = self._train_disc(self.G_ba, self.D_ba, self.D_ba_optimizer, img_b)
        loss = (loss_a + loss_b) * 0.5

        return loss

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
            images.append(recon_b[0].cpu().numpy())
            images.append(iden_b[0].cpu().numpy())

            images.append(img_b[0].cpu().numpy())
            images.append(fake_a[0].cpu().numpy())
            images.append(recon_a[0].cpu().numpy())
            images.append(iden_a[0].cpu().numpy())

        return np.array(images)

    def get_checkpoint(self):
        model = {'G_ab': self.G_ab.state_dict(),
                 'D_ab': self.D_ab.state_dict(),
                 'G_ab_optimizer': self.G_ab_optimizer.state_dict(),
                 'D_ab_optimizer': self.D_ab_optimizer.state_dict(),
                 'G_ba': self.G_ba.state_dict(),
                 'D_ba': self.D_ba.state_dict(),
                 'G_ba_optimizer': self.G_ba_optimizer.state_dict(),
                 'D_ba_optimizer': self.D_ba_optimizer.state_dict()}
        return model

    def load_checkpoint(self, checkpoint):
        self.G_ab.load_state_dict(checkpoint['G_ab'])
        self.D_ab.load_state_dict(checkpoint['D_ab'])
        self.G_ab_optimizer = torch.optim.Adam(self.G_ab.parameters())
        self.D_ab_optimizer = torch.optim.Adam(self.D_ab.parameters())
        self.G_ab_optimizer.load_state_dict(checkpoint['G_ab_optimizer'])
        self.D_ab_optimizer.load_state_dict(checkpoint['D_ab_optimizer'])

        self.G_ba.load_state_dict(checkpoint['G_ba'])
        self.D_ba.load_state_dict(checkpoint['D_ba'])
        self.G_ba_optimizer = torch.optim.Adam(self.G_ba.parameters())
        self.D_ba_optimizer = torch.optim.Adam(self.D_ba.parameters())
        self.G_ba_optimizer.load_state_dict(checkpoint['G_ba_optimizer'])
        self.D_ba_optimizer.load_state_dict(checkpoint['D_ba_optimizer'])



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