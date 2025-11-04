import torch
import torch.nn as nn
import torch.optim as optim

from models.valina_gan import GAN


class Discriminator(nn.Module):
    def __init__(self, num_classes = 0):
        super().__init__()
        self.num_classes = num_classes
        self.label = nn.Sequential(
            nn.Linear(self.num_classes, 28 * 28),
            nn.LeakyReLU(0.2),
        )

        input_channels = 1
        if self.num_classes > 0:
            input_channels = 2

        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
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
            nn.Linear(7 * 7 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        if self.num_classes > 0:
            label = self.label(label.float())
            label = label.view(*x.size())
            x = torch.concat((x, label), dim=1)

        x = self.feature(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, num_classes = 0):
        super().__init__()
        self.num_classes = num_classes
        # self.input = nn.Sequential(
        #     nn.Linear(100 + num_classes, 3136),
        #     nn.BatchNorm1d(3136),
        #     nn.ReLU()
        # )

        self.feature = nn.Sequential(
            nn.ConvTranspose2d(100 + num_classes, 100 + num_classes, kernel_size=7, stride=1, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(100 + num_classes),
            nn.LeakyReLU(0.2),

            # 7 -> 14
            nn.ConvTranspose2d(100 + num_classes, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 28 -> 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        # x = self.input(x)
        # x = x.view(x.size(0), 64, 7, 7)
        return self.feature(x)

class DCGAN_B(GAN):

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator(num_classes).to(device)
        self.D = Discriminator(num_classes).to(device)

        if is_train:
            self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = nn.BCELoss()


    def train_generator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        self.G.train()

        fake_x = self.G(self._generate_seed(y))

        onehot = None
        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(y, self.num_classes).to(self.device)

        pred_fake = self.D(fake_x, onehot)
        loss = self.criterion(pred_fake, one)
        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()

        return loss.item()

    def train_discriminator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        zero = torch.zeros((y.size(0), 1)).to(self.device)

        x = x.to(self.device)

        self.D.train()
        onehot = None

        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(y, self.num_classes).to(self.device)

        pred_real = self.D(x, onehot)
        real_loss = self.criterion(pred_real, one)
        self.D_optimizer.zero_grad()
        real_loss.backward()
        self.D_optimizer.step()

        fake_x = self.G(self._generate_seed(y)).detach()

        pred_fake = self.D(fake_x, onehot)
        fake_loss = self.criterion(pred_fake, zero)
        self.D_optimizer.zero_grad()
        fake_loss.backward()
        self.D_optimizer.step()

        return real_loss.item(), fake_loss.item()

    def _generate_seed(self, labels):
        seed = torch.randn(labels.size(0), 100).to(self.device)
        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(labels, self.num_classes).to(self.device)
            seed = torch.cat((seed, onehot), dim=1)
        seed = seed.reshape(labels.size(0), 100 + self.num_classes, 1, 1)
        return seed


if __name__ == '__main__':
    gen = Generator(10)
    seed = torch.randn((2, 110, 1, 1))
    images = gen(seed)
    print(images.shape)

    onehot = torch.randn(2, 10)
    dis = Discriminator(10)
    pred = dis(images, onehot)
    print(pred.shape)
