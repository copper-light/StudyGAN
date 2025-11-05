import os

from fontTools.misc.plistlib import end_data

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import torch
import logging
import numpy as np
import time
import argparse

from datetime import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils.util import show_plt, str2bool
from models.valina_gan import GAN
from models.dcgan import DCGAN
from models.dcgan_b import DCGAN_B
from models.wgan import WGAN
from models.wgan_gp import WGAN_GP
from models.wgan_gp_c import WGAN_GP_C

TIME_FORMAT = ('%Y%m%d_%H%M%S')

class StreamFlushingHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

out = StreamFlushingHandler(sys.stdout)
out.setLevel(logging.DEBUG)
out.addFilter(lambda record: record.levelno <= logging.INFO)  # DEBUG or INFO only

err = StreamFlushingHandler(sys.stderr)
err.setLevel(logging.WARNING) # WARNING and higher

logger = logging.getLogger(__name__)
logger.addHandler(out)
logger.addHandler(err)

class Trainer:
    def __init__(self, model, iter_per_print = 100, train_gen_per_iter = 5, gp_weight = 10, log_path="log"):
        self.model = model
        self.gp_weight = gp_weight
        self.train_gen_per_iter = train_gen_per_iter
        self.iter_per_print = iter_per_print
        self.step = 0
        self.losses = {'G':[], 'D':[], 'GP':[]}
        self.log_path = os.path.join(os.path.join(log_path, self.model.name), f'expr_{datetime.fromtimestamp(time.time()).strftime(TIME_FORMAT)}')
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join(self.log_path, 'train.log'), level=logging.INFO)

        self.writer = SummaryWriter(self.log_path)

    def _epoch(self, epoch, dataloader):
        g_losses = []
        d_losses = []
        start_time = time.time()
        progress = tqdm(enumerate(dataloader))
        for step, (x, y) in progress:
            self.step += 1
            d_loss = self.model.train_discriminator(x, y)
            d_losses.append(d_loss)
            if step % self.train_gen_per_iter == 0:
                g_loss = self.model.train_generator(x, y)
                g_losses.append(g_loss)

            progress.set_postfix({'epoch':epoch + 1, 'step': self.step, 'd_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses)})

        classes = 10
        if self.model.num_classes > 0:
            classes = self.model.num_classes

        classes = [[i for _ in range(classes)] for i in range(classes)]
        classes = torch.tensor(classes).to(self.model.device).reshape(-1)
        
        # classes = torch.tensor(list(range(classes))).to(self.model.device)
        
        images = self.model.generate_image_to_numpy(classes)
        
        save_image = show_plt(images, n_rows=10, n_cols=10, show=False, save_path=os.path.join(self.log_path, f'{self.model.name}_image_epoch_{epoch+1}.png'))
        save_image = torch.tensor(save_image).permute(2,0,1).unsqueeze(dim=0)
        
        self.losses['G'].append(np.mean(g_losses))
        self.losses['D'].append(np.mean(d_losses))

        s = int(time.time() - start_time)
        m = s // 60
        s = s % 60
        logging.info(f"elapsed: {m:02d}:{s:02d}, epoch: {epoch + 1}, step: {self.step}, d_loss: {np.mean(d_losses):.04f}, g_loss: {np.mean(g_losses):.04f}")

        self.writer.add_scalar('d_loss', np.mean(d_losses), epoch)
        self.writer.add_scalar('g_loss', np.mean(g_losses), epoch)
        self.writer.add_images('image', save_image, epoch)

    def train(self, dataloader, num_epochs):
        logging.info("params: ")
        for name, value in self.model.__dict__.items():
            logging.info(f"{name}: {value}")

        for name, value in self.__dict__.items():
            logging.info(f"{name}: {value}")

        logging.info("training.. ")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self._epoch(epoch, dataloader)

            if best_loss > self.losses['G'][-1]:
                best_loss = self.losses['G'][-1]
                model_chk = self.model.get_checkpoint()
                checkpoint = {'epoch': epoch + 1, 'step': self.step, 'loss': self.losses, 'model': model_chk}
                torch.save(checkpoint, os.path.join(self.log_path, f'{self.model.name}_checkpoint_epoch_{epoch+1}.pth'))
                logging.info(f"saved model - epoch:{epoch}, best_loss:{best_loss}")

        model_chk = self.model.get_checkpoint()
        checkpoint = {'epoch': num_epochs, 'step': self.step, 'loss': self.losses, 'model': model_chk}
        torch.save(checkpoint, os.path.join(self.log_path, f'{self.model.name}_checkpoint_epoch_{num_epochs}_latest.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN model trainer.")
    parser.add_argument('--models', type=str, default='DCGAN', choices=['GAN', 'GAN-C', 'DCGAN', 'DCGAN-C', 'WGAN', 'WGAN-GP','WGAN-GP-C'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--use-gpu', type=str2bool, default=True, choices=['True', 'False', 'true', 'false'])
    parser.add_argument('--log-path', type=str, default='logs/')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    args = parser.parse_args()

    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from math import prod

    train_dataset = None
    dataloader = None
    data_shape = None

    if args.dataset == 'mnist':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        data_shape = (1, 28, 28)

    elif args.dataset == 'cifar10':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        train_dataset = datasets.CIFAR10(root="data/", train=True, download=True, transform=transforms)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        data_shape = (3, 32, 32)

    assert dataloader is not None, "Not found dataset"

    device = 'cpu'
    if args.use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'

    if args.models == 'GAN':
        model = GAN(input_dim=data_shape, output_dim=data_shape, name="GAN", device=device, is_train=True, lr=args.lr)
        train_gen_per_iter = 1
        trainer = Trainer(model, train_gen_per_iter = train_gen_per_iter, log_path = args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'GAN-C':
        model = GAN(input_dim=data_shape, output_dim=data_shape, name="GAN-Conditional", device=device, is_train=True, lr=args.lr, num_classes=len(train_dataset.classes))
        train_gen_per_iter = 1
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path = args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'DCGAN':
        model = DCGAN(input_dim=data_shape, output_dim=data_shape, name="DCGAN", device=device, is_train=True, lr=args.lr)
        train_gen_per_iter = 1
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'DCGAN-C':
        model = DCGAN(input_dim=data_shape, output_dim=data_shape, name="DCGAN-Conditional", device=device, is_train=True, lr=args.lr, num_classes=len(train_dataset.classes))
        train_gen_per_iter = 1
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'WGAN':
        clip_threshold = 0.1
        train_gen_per_iter = 5
        model = WGAN(input_dim=data_shape, output_dim=data_shape, name="WGAN", device=device, is_train=True, lr=args.lr, clip_threshold = clip_threshold)
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'WGAN-GP':
        train_gen_per_iter = 5
        gp_weight = 10
        model = WGAN_GP_C(input_dim=data_shape, output_dim=data_shape, name="WGAN-GP", device=device, is_train=True, lr=args.lr, gp_weight=gp_weight)
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path)
        trainer.train(dataloader, args.epochs)

    elif args.models == 'WGAN-GP-C':
        train_gen_per_iter = 5
        gp_weight = 10
        model = WGAN_GP_C(input_dim=data_shape, output_dim=data_shape, 
                          name="WGAN-GP", device=device, is_train=True, 
                          lr=args.lr, gp_weight=gp_weight, 
                          num_classes=len(train_dataset.classes))
        trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path)
        trainer.train(dataloader, args.epochs)
