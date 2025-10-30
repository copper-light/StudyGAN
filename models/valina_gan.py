import torch
from torch import nn
from models.model import Model

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes = 0):
        super().__init__()

        in_features = input_dim + num_classes
        self.fc = nn.Sequential(
            nn.Linear(in_features, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Dropout(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, output_dim, num_classes = 0):
        super().__init__()

        self.gen_image = nn.Sequential(
            nn.Linear(100 + num_classes, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.gen_image(x)
        return x


class GAN(Model):
    def __init__(self, input_dim, output_dim, name="GAN", num_classes = 0, device = 'cpu', is_train = True, lr=1e-4):
        super().__init__(input_dim, output_dim, name, num_classes, device, is_train, lr)

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.G = Generator(output_dim, num_classes).to(device)
        self.D = Discriminator(input_dim, num_classes).to(device)

        if is_train:
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = nn.BCELoss()

    def train_generator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        self.G.train()

        fake_x = self.G(self._generate_seed(y))
        fake_x = fake_x.reshape(fake_x.size(0), self.input_dim)

        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(y, self.num_classes).to(self.device)
            fake_x = torch.cat((fake_x, onehot), dim=1)

        pred_fake = self.D(fake_x)
        loss = self.criterion(pred_fake, one)
        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()

        return loss.item()

    def train_discriminator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        zero = torch.zeros((y.size(0), 1)).to(self.device)

        x = x.reshape(x.size(0), self.input_dim).to(self.device)

        self.D.train()
        onehot = None

        if self.num_classes > 0:
            onehot = torch.nn.functional.one_hot(y, self.num_classes).to(self.device)
            x = torch.cat((x, onehot), dim=1)

        pred_real = self.D(x)
        real_loss = self.criterion(pred_real, one)
        self.D_optimizer.zero_grad()
        real_loss.backward()
        self.D_optimizer.step()

        fake_x = self.G(self._generate_seed(y)).detach()
        fake_x = fake_x.reshape(fake_x.size(0), self.input_dim)

        if self.num_classes > 0:
            fake_x = torch.cat((fake_x, onehot), dim=1)

        pred_fake = self.D(fake_x)
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
        return seed

    def generate_image_to_numpy(self, classes):
        images = None
        self.G.eval()
        with torch.no_grad():
            images = self.G(self._generate_seed(classes))
            images = images.reshape(-1, 1, 28, 28).cpu().numpy()
        return images

    def get_checkpoint(self):
        model = {'G': self.G.state_dict(),
                 'D': self.D.state_dict(),
                 'G_optimizer': self.G_optimizer.state_dict(),
                 'D_optimizer': self.D_optimizer.state_dict()}
        return model

    def load_checkpoint(self, checkpoint):
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer = torch.optim.Adam(self.G.parameters())
        self.D_optimizer = torch.optim.Adam(self.D.parameters())
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])


if __name__ == "__main__":
    gan = GAN(input_dim=(28*28), output_dim= (28*28))
    labels = torch.tensor([0, 1])
    images = gan.generate_image_to_numpy(labels)

    # import matplotlib.pyplot as plt
    # plt.imshow(images[0])
    print(images.shape)
