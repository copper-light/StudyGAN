import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

# https://github.com/EmilienDupont/wgan-gp/blob/master/training.py

def createSeed(class_indexes, classes_num=None, output_dim=100):
    seed = torch.randn(len(class_indexes), output_dim)
    if classes_num is not None:
        onehot = nn.functional.one_hot(class_indexes, classes_num)
        seed = torch.concat([seed, onehot], dim=1)
    return seed

def gradient_penalty(pred, img, device):
    gradients = torch.autograd.grad(outputs = pred,
                                    inputs = img,
                                    grad_outputs=torch.ones(pred.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(pred.size(0), -1)
    l2_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return torch.mean((1 - l2_norm) ** 2)


def randomWeightedAverage(a, b, batch_size, device):
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * a.detach()) + ((1 - alpha) * b.detach())
    # interpolated = torch.tensor(interpolated, requires_grad=True)
    return interpolated


def show_plt(images, col_size, save_path = None):
    fig, axes = plt.subplots(1, col_size, figsize=(15, 6))  # 2행 5열 격자 생성

    for i in range(col_size):
        ax = axes[i]
        image = images[i].reshape(28, 28).cpu()

        # 예시로 각 그림에 숫자 표시
        ax.imshow(image.detach().numpy(), cmap='gray')
        ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    if save_path != None:
        plt.savefig(save_path)

    plt.show()


class Trainer():
    def __init__(self, generator, critic, g_optimizer, d_optimizer, device, train_gen_per_iter = 5, gp_weight = 10):
        self.G = generator.to(device)
        self.D = critic.to(device)
        self.G_optimizer = g_optimizer
        self.D_optimizer = d_optimizer
        self.device = device
        self.progress = None
        self.gp_weight = gp_weight
        self.train_gen_per_iter = train_gen_per_iter
        self.step = 0

    def _iter_g(self, x, y):
        self.G.train()
        self.G.zero_grad()
        seed = createSeed(y).to(self.device)
        fake_x = self.G(seed).detach()
        pred = self.D(fake_x)

        loss = -torch.mean(pred)
        loss.backward()
        self.G_optimizer.step()
        return loss.item()

    def _iter_d(self, x, y):
        batch_size = x.shape[0]
        self.D.train()
        self.D.zero_grad()

        x = x.to(self.device)
        real_pred = self.D(x)

        seed = createSeed(y).to(self.device)
        fake_x = self.G(seed).detach()
        fake_pred = self.D(fake_x)

        inter_x = randomWeightedAverage(x, fake_x, batch_size, self.device)
        inter_pred = self.D(inter_x)
        penalty_loss = gradient_penalty(inter_pred, inter_x, self.device)

        loss = torch.mean(real_pred) - torch.mean(fake_pred) + (self.gp_weight * penalty_loss)

        loss.backward()
        self.D_optimizer.step()

        return loss.item()

    def _epoch(self, epoch, dataloader):
        g_losses = []
        d_losses = []
        for step, (x, y) in enumerate(dataloader):
            self.step += 1
            x = x.to(self.device)
            d_loss = self._iter_d(x, y)
            d_losses.append(d_loss)
            if step % self.train_gen_per_iter == 0:
                g_loss = self._iter_g(x, y)
                g_losses.append(g_loss)

            self.progress.set_postfix({'step': self.step, 'd_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses)})

    def train(self, dataloader, num_epochs,  log_path="log"):
        num_classes = len(dataloader.dataset.classes)
        self.progress = tqdm(range(num_epochs))
        for epoch in self.progress:
            self._epoch(epoch, dataloader)

            classes = torch.tensor([v for v in range(num_classes)])
            seed = createSeed(classes).to(device)
            images = self.G(seed)
            show_plt(images, num_classes, f'{log_path}/checkpoint_{epoch}_last.png')

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from WGANGP import Generator, Discriminator

    lr = 1e-4
    batch_size = 64
    epochs = 40

    g = Generator()
    d = Discriminator()

    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr)

    train_dataset = datasets.MNIST(root="../data/",
                                   train=True,
                                   transform=transforms.ToTensor())

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    trainer = Trainer(g, d, g_optimizer, d_optimizer, device, gp_weight= 10)
    trainer.train(dataloader, epochs, log_path = 'log/')