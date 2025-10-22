import logging
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import numpy as np

# from CnnGAN import show_plt, createOnehot2DSeed
from GAN import Discriminator, Generator, createOnehotSeed, show_plt


logging.basicConfig(filename='log/train.log', level=logging.INFO)

device = 'cpu'
# mps에서는 conv_transpose2d 구현되지 않음
# if torch.mps.is_available():
#     device = 'mps'

if torch.cuda.is_available():
    device = 'cuda'

print(device)


if __name__ == "__main__":
    lr = 0.0001
    num_epochs = 20

    train_dataset = datasets.MNIST(root = "../data/",
                               train = True,
                               transform = transforms.ToTensor())

    loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

    num_classes = len(train_dataset.classes)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion = torch.nn.BCELoss()
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    genr_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    
    def train_step(model, x, target, criterion, optimizer):
        model.train()
        pred = model(x)
        pred = pred.squeeze(dim=0)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
        
    bestest = float('inf')
    progress = tqdm(range(1, num_epochs + 1))

    avg_gen_loss = None
    avg_dis_loss = None
    for epoch in progress:
        dis_losses = []
        gen_losses = []
    
        start_time = time.time()
        
        for step, (x, label) in enumerate(loader):
            batch_size = x.shape[0]
            x = x.to(device)
            target = torch.ones(batch_size, 1).to(device)
            pos_loss = train_step(discriminator, x, target, criterion, disc_optimizer)
    
            seed = createOnehotSeed(torch.ones(int(batch_size)), num_classes).to(device)
            x = generator(seed).detach()
            target = torch.zeros(int(batch_size), 1).to(device)
            neg_loss = train_step(discriminator, x, target, criterion, disc_optimizer)
    
            seed = createOnehotSeed(label.reshape(-1), num_classes).to(device)
            x = generator(seed)
            target = torch.ones(batch_size, 1).to(device)
            gen_loss = train_step(discriminator, x, target, criterion, genr_optimizer)
    
            dis_losses.append(pos_loss)
            dis_losses.append(neg_loss)
            gen_losses.append(gen_loss)
    
            progress.set_postfix_str(f"{step + 1}/{len(loader)}, dis_loss: {np.mean(dis_losses):.04f}, gen_loss: {np.mean(gen_losses):.04f}")
            if step % 10000 == 0:
                show_plt(generator, num_classes, f'log/checkpoint_{epoch}_{step}.png')
                logging.info(f'---- Step {step}, DiscLoss: {np.mean(dis_losses):.04f}, GenLoss: {np.mean(gen_losses):.04f}')

        elapsed = time.time() - start_time
        avg_gen_loss = np.mean(gen_losses)
        avg_dis_loss = np.mean(dis_losses)

        show_plt(generator, num_classes, f'log/checkpoint_{epoch}_{step}.png')
        logging.info(f'Epoch {epoch}, elapsed: {elapsed}, DiscLoss: {avg_dis_loss:.04f}, GenLoss: {avg_gen_loss:.04f}')

        if bestest > avg_gen_loss:
            checkpoint = {
                'epoch': epoch,
                'gen_model_state_dict': generator.state_dict(),
                'disc_model_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': genr_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'disc_loss': avg_dis_loss,
                'gen_loss': avg_gen_loss,
            }
            torch.save(checkpoint, f'log/checkpoint_{epoch}.pth')
            bestest = avg_gen_loss

    checkpoint = {
        'epoch': num_epochs,
        'gen_model_state_dict': generator.state_dict(),
        'disc_model_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': genr_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'disc_loss': avg_dis_loss,
        'gen_loss': avg_gen_loss,
    }
    torch.save(checkpoint, f'log/latest.pth')

    generator.eval()
    show_plt(generator, 10)