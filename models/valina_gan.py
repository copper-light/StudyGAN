import torch
from torch import nn
from models.model import Model

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 200),
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
    def __init__(self):
        super().__init__()

        self.gen_image = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.gen_image(x)
        return x


class GAN(Model):
    def __init__(self, name="GAN", device = 'cpu', is_train = True, lr=1e-4):
        super().__init__(name, device)
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)

        if is_train:
            self.G_Optimizer = torch.optim.Adam(self.G.parameters(), lr=lr)
            self.D_Optimizer = torch.optim.Adam(self.D.parameters(), lr=lr)
            self.criterion = nn.BCELoss()

    def train_generator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        self.G.train()

        fake_x = self.G(self._generate_seed(y))
        pred_fake = self.D(fake_x)
        loss = self.criterion(pred_fake, one)
        self.G_Optimizer.zero_grad()
        loss.backward()
        self.G_Optimizer.step()

        return loss.item()

    def train_discriminator(self, x, y):
        one = torch.ones((y.size(0), 1)).to(self.device)
        zero = torch.zeros((y.size(0), 1)).to(self.device)
        x = x.reshape(x.size(0), -1).to(self.device)
        self.D.train()

        pred_real = self.D(x)
        real_loss = self.criterion(pred_real, one)
        self.D_Optimizer.zero_grad()
        real_loss.backward()
        self.D_Optimizer.step()

        fake_x = self.G(self._generate_seed(y)).detach()
        pred_fake = self.D(fake_x)
        fake_loss = self.criterion(pred_fake, zero)
        self.D_Optimizer.zero_grad()
        fake_loss.backward()
        self.D_Optimizer.step()

        return real_loss.item(), fake_loss.item()

    def _generate_seed(self, labels):
        return torch.randn(labels.size(0), 100).to(self.device)

    def generate_image_to_numpy(self, num_images):
        images = None
        self.G.eval()
        with torch.no_grad():
            labels = torch.tensor(list(range(num_images)))
            images = self.G(self._generate_seed(labels))
            images = images.reshape(-1, 1, 28, 28).cpu().numpy()
        return images

    def get_checkpoint(self):
        model = {'G': self.G.state_dict(),
                 'D': self.D.state_dict(),
                 'G_Optimizer': self.G_Optimizer.state_dict(),
                 'D_Optimizer': self.D_Optimizer.state_dict()}
        return model

    def load_checkpoint(self, checkpoint):
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_Optimizer = torch.optim.Adam(self.G.parameters())
        self.D_Optimizer = torch.optim.Adam(self.D.parameters())
        self.G_Optimizer.load_state_dict(checkpoint['G_Optimizer'])
        self.D_Optimizer.load_state_dict(checkpoint['D_Optimizer'])


if __name__ == "__main__":
    gan = GAN()
    labels = torch.tensor([0, 1])
    images = gan.generate_image(labels)

    import matplotlib.pyplot as plt
    plt.imshow(images[0])
    print(images.shape)
