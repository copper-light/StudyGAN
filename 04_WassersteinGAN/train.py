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
    gradients = torch.autograd.grad(outputs = pred, inputs = img,
                                    grad_outputs=torch.ones(pred.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(pred.size(0), -1)
    l2_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
    return torch.mean((l2_norm - 1) ** 2)


def randomWeightedAverage(real_x, fake_x, device):
    batch_size = real_x.size()[0]
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_x)
    interpolated = (alpha * real_x.data) + ((1 - alpha) * fake_x.data)
    interpolated.requires_grad_(True)
    return interpolated


def show_plt(images, col_size, save_path = None):
    fig, axes = plt.subplots(1, col_size, figsize=(15, 6))  # 2행 5열 격자 생성

    for i in range(col_size):
        ax = axes[i]
        image = images[i].reshape(28, 28).cpu()
        image = image * 0.5 + 0.5

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
        self.gp_weight = gp_weight
        self.train_gen_per_iter = train_gen_per_iter
        self.step = 0
        self.losses = {'G':[], 'D':[], 'GP':[]}

    def _iter_g(self, x, y):
        self.G.train()
        seed = createSeed(y).to(self.device)
        fake_x = self.G(seed)
        pred = self.D(fake_x)

        loss = -torch.mean(pred)
        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()
        return loss.item()

    def _iter_d(self, x, y):
        self.D.train()
        self.D_optimizer.zero_grad()

        x = x.to(self.device)
        real_pred = self.D(x)

        seed = createSeed(y).to(self.device)
        fake_x = self.G(seed).detach()
        fake_pred = self.D(fake_x)

        inter_x = randomWeightedAverage(x, fake_x, self.device)
        inter_pred = self.D(inter_x)
        penalty_loss = gradient_penalty(inter_pred, inter_x, self.device)

        loss = fake_pred.mean() - real_pred.mean() + (self.gp_weight * penalty_loss)
        loss.backward()
        self.D_optimizer.step()

        self.losses['GP'].append(penalty_loss.item())

        return loss.item()

    def _epoch(self, epoch, dataloader):
        g_losses = []
        d_losses = []

        progress = tqdm(enumerate(dataloader))
        for step, (x, y) in progress:
            self.step += 1
            x = x.to(self.device)
            d_loss = self._iter_d(x, y)
            d_losses.append(d_loss)
            if step % self.train_gen_per_iter == 0:
                g_loss = self._iter_g(x, y)
                g_losses.append(g_loss)

            gp_loss = self.losses['GP'][-100:]
            progress.set_postfix({'epoch':epoch + 1, 'step': self.step, 'd_loss': np.mean(d_losses), 'gp_loss': np.mean(gp_loss), 'g_loss': np.mean(g_losses)})

    def train(self, dataloader, num_epochs,  log_path="log"):
        num_classes = len(dataloader.dataset.classes)

        for epoch in range(num_epochs):
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
    betas = (.5, .999)
    batch_size = 32
    epochs = 80

    g = Generator()
    d = Discriminator()

    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr, betas=betas)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr, betas=betas)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = datasets.MNIST(root="../data/", train=True, transform=transforms)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    trainer = Trainer(g, d, g_optimizer, d_optimizer, device, gp_weight= 10)
    trainer.train(dataloader, epochs, log_path = 'log/')