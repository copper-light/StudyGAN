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
from models.cycle_gan_unet import CycleGAN
from models.cycle_gan_resnet import CycleGANResNet

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
    def __init__(self, model, iter_per_print = 100, train_gen_per_iter = 5, gp_weight = 10, log_path="log", checkpoint=None):
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

        self.checkpoint = checkpoint
        self.start_epoch = 1

        if checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
            model.load_checkpoint(checkpoint['model'])
            self.start_epoch = checkpoint['epoch'] + 1

    def _epoch(self, epoch, train_loader, valid_loader):
        g_losses = []
        d_losses = []
        monitor_values = {}
        start_time = time.time()
        progress = tqdm(enumerate(train_loader))

        if hasattr(train_loader.dataset, "on_epoch_start"):
            train_loader.dataset.on_epoch_start()
        
        for step, (x, y) in progress:
            self.step += 1
            d_loss, d_values = self.model.train_discriminator(x, y)
            d_losses.append(d_loss)

            if d_values is not None:
                for name, value in d_values.items():
                    if name not in monitor_values:
                        monitor_values[name] = []
                    monitor_values[name].append(value)

            if step % self.train_gen_per_iter == 0:
                g_loss, g_values = self.model.train_generator(x, y)
                if g_values is not None:
                    for name, value in g_values.items():
                        if name not in monitor_values:
                            monitor_values[name] = []
                        monitor_values[name].append(value)
                g_losses.append(g_loss)

            progress.set_postfix({'epoch':epoch, 'step': self.step, 'd_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses)})

        self._valid(epoch, valid_loader)

        self.losses['G'].append(np.mean(g_losses))
        self.losses['D'].append(np.mean(d_losses))

        s = int(time.time() - start_time)
        m = s // 60
        s = s % 60
        logging.info(f"elapsed: {m:02d}:{s:02d}, epoch: {epoch}, step: {self.step}, d_loss: {np.mean(d_losses):.04f}, g_loss: {np.mean(g_losses):.04f}")
        for name, value in monitor_values.items():
            logging.info(f"  -{name}: {np.mean(value):.04f}")
            self.writer.add_scalar(name, np.mean(value), epoch)

        self.writer.add_scalar('d_loss', np.mean(d_losses), epoch)
        self.writer.add_scalar('g_loss', np.mean(g_losses), epoch)


    def _valid(self, epoch, dataloader):
        images = []
        for x, y in dataloader:
            images.append(self.model.generate_image_to_numpy(x, y))

        images = np.array(images)
        save_image = show_plt(images, n_rows=images.shape[0], n_cols=images.shape[1], show=False, save_path=os.path.join(self.log_path, f'{self.model.name}_image_epoch_{epoch+1}.png'))
        save_image = torch.tensor(save_image).permute(2,0,1).unsqueeze(dim=0)
        self.writer.add_images('image', save_image, epoch)

    def train(self, train_loader, valid_loader, num_epochs):
        logging.info("params: ")
        for name, value in self.model.__dict__.items():
            logging.info(f"{name}: {value}")

        for name, value in self.__dict__.items():
            logging.info(f"{name}: {value}")

        logging.info("training.. ")

        best_loss = float('inf')

        for epoch in range(self.start_epoch, num_epochs+1):
            self._epoch(epoch, train_loader, valid_loader)

            if best_loss > self.losses['G'][-1]:
                best_loss = self.losses['G'][-1]
                model_chk = self.model.get_checkpoint()
                checkpoint = {'epoch': epoch, 'step': self.step, 'loss': self.losses, 'model': model_chk}
                torch.save(checkpoint, os.path.join(self.log_path, f'{self.model.name}_checkpoint_epoch_{epoch}.pth'))
                logging.info(f"saved model - epoch:{epoch}, best_loss:{best_loss}")

        model_chk = self.model.get_checkpoint()
        checkpoint = {'epoch': num_epochs, 'step': self.step, 'loss': self.losses, 'model': model_chk}
        torch.save(checkpoint, os.path.join(self.log_path, f'{self.model.name}_checkpoint_epoch_{num_epochs}_latest.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN model trainer.")
    parser.add_argument('--models', type=str, default='DCGAN', choices=['GAN', 'GAN-C', 'DCGAN', 'DCGAN-C', 'WGAN', 'WGAN-GP','WGAN-GP-C', 'CYCLE-GAN', 'CYCLE-GAN-RESNET'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--use-gpu', type=str2bool, default=True, choices=['True', 'False', 'true', 'false'])
    parser.add_argument('--log-path', type=str, default='logs/')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'apple2orange', 'monet2photo'])
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    from math import prod

    train_dataset = None
    train_loader = None
    valid_dataset = None
    valid_loader = None

    dataloader = None
    data_shape = None

    model = None

    start_epoch = 1
    train_gen_per_iter = 1

    if args.dataset == 'mnist':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

        train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        classes = 10
        classes = [[(i, i) for _ in range(classes)] for i in range(classes)]
        valid_dataset = torch.tensor(classes).reshape(-1)
        valid_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

        data_shape = (1, 28, 28)

    elif args.dataset == 'cifar10':
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        train_dataset = datasets.CIFAR10(root="data/", train=True, download=True, transform=transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        classes = 10
        classes = [[(i, i) for _ in range(classes)] for i in range(classes)]
        valid_dataset = torch.tensor(classes).reshape(-1)
        valid_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)

        data_shape = (3, 32, 32)

    elif args.dataset == 'apple2orange': # apple2orange
        from datasets.styletransfer import StyleTransferDataset

        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = StyleTransferDataset(root="data/apple2orange/", train=True, transform=transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = StyleTransferDataset(root="data/apple2orange/", limit=10, train=False, transform=transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        data_shape = (3, 128, 128)

    elif args.dataset == 'monet2photo': # apple2orange
        from datasets.styletransfer import StyleTransferDataset

        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = StyleTransferDataset(root="data/monet2photo/", train=True, transform=transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = StyleTransferDataset(root="data/monet2photo/", limit=10, train=False, transform=transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

        data_shape = (3, 256, 256)


    assert train_loader is not None, "Not found dataset"

    device = 'cpu'
    if args.use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'

    if args.models == 'GAN':
        model = GAN(input_dim=data_shape, output_dim=data_shape, name="GAN", device=device, is_train=True, lr=args.lr)

    elif args.models == 'GAN-C':
        model = GAN(input_dim=data_shape, output_dim=data_shape, name="GAN-Conditional", device=device, is_train=True, lr=args.lr, num_classes=len(train_dataset.classes))

    elif args.models == 'DCGAN':
        model = DCGAN(input_dim=data_shape, output_dim=data_shape, name="DCGAN", device=device, is_train=True, lr=args.lr)

    elif args.models == 'DCGAN-C':
        model = DCGAN(input_dim=data_shape, output_dim=data_shape, name="DCGAN-Conditional", device=device, is_train=True, lr=args.lr, num_classes=len(train_dataset.classes))

    elif args.models == 'WGAN':
        clip_threshold = 0.1
        train_gen_per_iter = 5
        model = WGAN(input_dim=data_shape, output_dim=data_shape, name="WGAN", device=device, is_train=True, lr=args.lr, clip_threshold = clip_threshold)

    elif args.models == 'WGAN-GP':
        train_gen_per_iter = 5
        gp_weight = 10
        model = WGAN_GP(input_dim=data_shape, output_dim=data_shape, name="WGAN-GP", device=device, is_train=True, lr=args.lr, gp_weight=gp_weight)

    elif args.models == 'WGAN-GP-C':
        train_gen_per_iter = 5
        gp_weight = 10
        model = WGAN_GP_C(input_dim=data_shape, output_dim=data_shape, 
                          name="WGAN-GP", device=device, is_train=True, 
                          lr=args.lr, gp_weight=gp_weight, 
                          num_classes=len(train_dataset.classes))

    elif args.models == 'CYCLE-GAN':
        model = CycleGAN(input_dim=data_shape, output_dim=data_shape,
                         name="CYCLE-GAN", device=device, is_train=True,
                         lr=args.lr,
                         gen_n_filters=32, disc_n_filters=32,
                         lambda_validation = 1, lambda_reconstruction = 10, lambda_identity = 2)
    elif args.models == 'CYCLE-GAN-RESNET':
        model = CycleGANResNet(input_dim=data_shape, output_dim=data_shape,
                         name="CYCLE-GAN-RESNET", device=device, is_train=True,
                         lr=args.lr,
                         gen_n_filters=32, disc_n_filters=64,
                         lambda_validation=1, lambda_reconstruction=10, lambda_identity=5)

    trainer = Trainer(model, train_gen_per_iter=train_gen_per_iter, log_path=args.log_path, checkpoint=args.checkpoint)
    trainer.train(train_loader, valid_loader, args.epochs)