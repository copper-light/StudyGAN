import torch
import torch.nn as nn

from models.model import Model
from models.valina_gan import GAN

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # 14
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # 7
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1, 1)
            # nn.Sigmoid()
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
            nn.ReLU(True),

            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 28 -> 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), 64, 7, 7)
        return self.feature(x)


class WassersteinLoss(nn.Module):
    def forward(self, preds, targets):
        return -torch.mean(preds * targets)


class WGAN(GAN):
    def __init__(self, input_dim, output_dim, name="WGAN", num_classes = 0, device = 'cpu', is_train = True, lr=5e-5, gp_weight = 1, clip_threshold = 0.01):
        super().__init__(input_dim, output_dim, name, num_classes, device, is_train, lr)
        self.gp_weight = gp_weight
        self.clip_threshold = clip_threshold

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)

        if is_train:
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = WassersteinLoss()

    # def _generate_seed(self, labels):
    #     pass

    def train_discriminator(self, x, target):
        self.D.train()
        self.D_optimizer.zero_grad()

        real_target = torch.ones(x.size(0), 1).to(self.device)
        fake_target = -torch.ones(x.size(0), 1).to(self.device)

        x = x.to(self.device)
        real_output = self.D(x)
        real_loss = self.criterion(real_output, real_target)

        seed = self._generate_seed(target).to(self.device)
        fake_x = self.G(seed).detach()
        fake_output = self.D(fake_x)
        fake_loss = self.criterion(fake_output, fake_target)

        loss = fake_loss - real_loss
        loss.backward()
        self.D_optimizer.step()

        if self.clip_threshold != None:
            with torch.no_grad():
                for param in self.D.parameters():
                    param.data.clamp_(-self.clip_threshold, self.clip_threshold)

        return loss.item()

    def train_generator(self, x, target):
        real_target = torch.ones(x.size(0), 1).to(self.device)

        self.G.train()
        seed = self._generate_seed(target).to(self.device)
        fake_x = self.G(seed)
        output = self.D(fake_x)
        loss = self.criterion(output, real_target)
        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()
        return loss.item()

    def _gradient_penalty(self, pred, img):
        gradients = torch.autograd.grad(outputs=pred, inputs=img,
                                        grad_outputs=torch.ones(pred.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(pred.size(0), -1)
        l2_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
        return torch.mean((l2_norm - 1) ** 2)

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
    pass