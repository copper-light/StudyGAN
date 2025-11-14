import torch
import torch.nn as nn

from models.model import Model
from models.valina_gan import GAN
from models.dcgan import Generator as DCGAN_Generator
from models.dcgan import Discriminator as DCGAN_Discriminator

class Discriminator(nn.Module):

    def __init__(self, input_dim, num_classes=0):
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
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # 14
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            # 7
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(int((self.w * self.h) / 16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), self.in_channels, self.h, self.w)
        x = self.feature(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Generator(DCGAN_Generator):
    pass


class WassersteinLoss(nn.Module):
    def forward(self, preds, targets):
        return -torch.mean(preds * targets)

def gradient_penalty(pred, img, device):
    gradients = torch.autograd.grad(outputs=pred, inputs=img,
                                    grad_outputs=torch.ones(pred.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(pred.size(0), -1)
    l2_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
    return torch.mean((l2_norm - 1) ** 2)


class WGAN_GP_C(GAN):
    def __init__(self, input_dim, output_dim, name="WGAN-GP", num_classes = 0, device = 'cpu', is_train = True, lr=5e-5, gp_weight = 10):
        super().__init__(input_dim, output_dim, name, num_classes, device, is_train, lr)
        self.gp_weight = gp_weight

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator(output_dim, num_classes).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)

        if is_train:
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = WassersteinLoss()

    def train_discriminator(self, x, target):
        self.D.train()

        # real_target = torch.ones(x.size(0), 1).to(self.device)
        # fake_target = -torch.ones(x.size(0), 1).to(self.device)

        x = x.to(self.device)

        seed = self._generate_seed(target).to(self.device)
        fake_x = self.G(seed).detach()
        inter_x = self._generate_interpolated_image(x, fake_x)

        x = x.flatten(1)

        onehot = None
        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(target, self.num_classes).to(self.device)
            x = torch.cat((x, onehot), dim=1)

        real_output = self.D(x)
        real_loss = torch.mean(real_output)

        fake_x = fake_x.flatten(1)
        if self.num_classes > 0:
            fake_x = torch.cat((fake_x, onehot), dim=1)

        fake_output = self.D(fake_x)
        fake_loss =  torch.mean(fake_output)

        inter_output = None
        if self.num_classes > 0:
            inter_x_onehot = torch.cat((inter_x.flatten(1), onehot), dim=1)
            inter_output = self.D(inter_x_onehot)
        else:
            inter_output = self.D(inter_x.flatten(1))

        inter_loss = gradient_penalty(inter_output, inter_x, self.device)

        loss = torch.mean(fake_output) - torch.mean(real_output) + (inter_loss * self.gp_weight)

        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()

        wasserstein_loss = real_loss - fake_loss
        return loss.item(), {'wasserstein_loss': wasserstein_loss.item()}

    def train_generator(self, x, target):
        self.G.train()

        real_target = torch.ones(x.size(0), 1).to(self.device)

        seed = self._generate_seed(target).to(self.device)
        fake_x = self.G(seed)
        fake_x = fake_x.flatten(1)

        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(target, self.num_classes).to(self.device)
            fake_x = torch.cat((fake_x, onehot), dim=1)

        output = self.D(fake_x)
        loss = self.criterion(output, real_target)
        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()
        return loss.item(), None

    def _generate_interpolated_image(self, real_x, fake_x):
        batch_size = real_x.size()[0]
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_x)
        interpolated = (alpha * real_x.data) + ((1 - alpha) * fake_x.data)
        interpolated.requires_grad_(True)
        return interpolated

    # def generate_image_to_numpy(self, size):
    #     pass

    # def get_checkpoint(self):
    #     pass
    #
    # def load_checkpoint(self, checkpoint):
    #     pass


if __name__ == '__main__':
    model = WGAN_GP_C(input_dim=(3,32,32), output_dim=(3,32,32), num_classes=10)

    gen = Generator(output_dim=(3,32,32), num_classes=10)
    dis = Discriminator(input_dim=(3, 32, 32), num_classes=10)

    x = torch.randn((4, 100 + 10))
    output = gen(x)
    print("gen", output.shape)
    onehot = torch.randn(4,10)
    output = output.flatten(1)
    output = torch.concat((output, onehot), dim=1)
    pred = dis(output)

    print("dis", pred.shape)