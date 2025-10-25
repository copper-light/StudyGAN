import torch
from torch import nn
from tqdm import tqdm

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
    # l2_norm = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    l2_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((1 - l2_norm) ** 2)
    return gradient_penalty


def randomWeightedAverage(a, b, batch_size, device):
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = (alpha * a) + ((1 - alpha) * b)
    interpolated = torch.tensor(interpolated, requires_grad=True)
    return interpolated


class Trainer():
    def __init__(self, generator, critic, g_optimizer, d_optimizer, device):
        self.G = generator.to(device)
        self.D = critic.to(device)
        self.G_optimizer = g_optimizer
        self.D_optimizer = d_optimizer
        self.device = device
        self.progress = None

    def _iter_g(self, x, y):
        self.G.train()
        self.G.zero_grad()
        seed = createSeed(y).to(self.device)
        pred = self.G(seed)

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

        loss = torch.mean(real_pred) - torch.mean(fake_pred) + penalty_loss

        loss.backward()
        self.D_optimizer.step()

        return loss.item()

    def _epoch(self, epoch, dataloader):
        for step, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            d_loss = self._iter_d(x, y)
            g_loss = self._iter_g(x, y)
            self.progress.set_postfix({'d_loss': d_loss, 'g_loss': g_loss})

    def train(self, dataloader, num_epochs, log="log/"):
        self.progress = tqdm(range(num_epochs))
        for epoch in self.progress:
            self._epoch(epoch, dataloader)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from WGANGP import Generator, Discriminator

    lr = 1e-4
    batch_size = 64
    epochs = 1

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

    trainer = Trainer(g, d, g_optimizer, d_optimizer, device)
    trainer.train(dataloader, epochs, 'log')