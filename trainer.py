import torch
import os
import logs
import numpy as np
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.util import show_plt
from models.model import Model
from models.valina_gan import GAN



class Trainer:
    def __init__(self, model, iter_per_print = 100, train_gen_per_iter = 5, gp_weight = 10):
        self.model = model
        self.gp_weight = gp_weight
        self.train_gen_per_iter = train_gen_per_iter
        self.iter_per_print = iter_per_print
        self.step = 0
        self.losses = {'G':[], 'D':[], 'GP':[]}

    def _epoch(self, epoch, dataloader, log_path):
        g_losses = []
        d_losses = []

        progress = tqdm(enumerate(dataloader))
        for step, (x, y) in progress:
            self.step += 1
            d_loss = self.model.train_discriminator(x, y)
            d_losses.append(d_loss)
            if step % self.train_gen_per_iter == 0:
                g_loss = self.model.train_generator(x, y)
                g_losses.append(g_loss)

            progress.set_postfix({'epoch':epoch + 1, 'step': self.step, 'd_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses)})
            #
            # if self.step % self.iter_per_print == 0:
            #

        images = self.model.generate_image_to_numpy(10)
        show_plt(images, n_rows=1, n_cols=10, show=False, save_path=os.path.join(log_path, f'{self.model.name}_image_{self.step}.png'))

        self.losses['G'].append(np.mean(g_losses))
        self.losses['D'].append(np.mean(d_losses))


    def train(self, dataloader, num_epochs, log_path="log"):
        num_classes = len(dataloader.dataset.classes)
        best_loss = float('inf')
        for epoch in range(num_epochs):
            self._epoch(epoch, dataloader, log_path)

            if best_loss > self.losses['G'][-1]:
                best_loss = self.losses['G'][-1]
                model_chk = self.model.get_checkpoint()
                checkpoint = {'epoch': epoch + 1, 'step': self.step, 'loss': self.losses, 'model': model_chk}

                os.makedirs(log_path, exist_ok=True)
                torch.save(checkpoint, os.path.join(log_path, f'{self.model.name}_checkpoint_epoch_{epoch}.pth'))

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    # from WGANGP import Generator, Discriminator

    lr = 1e-4
    betas = (.5, .999)
    batch_size = 32
    epochs = 3

    transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # GAN
    model = GAN(name="GAN", device=device, is_train=True, lr=lr)
    train_gen_per_iter = 1
    batch_size = 1

    trainer = Trainer(model, train_gen_per_iter = train_gen_per_iter)
    trainer.train(dataloader, epochs, log_path = 'logs/')