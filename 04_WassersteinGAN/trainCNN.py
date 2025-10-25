import logging
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import numpy as np

from CnnGAN import WassersteinLoss, Discriminator, Generator, createOnehotSeed, show_plt


logging.basicConfig(filename='log/train.log', level=logging.INFO)

device = 'cpu'
# mps에서는 conv_transpose2d 구현되지 않음
# if torch.mps.is_available():
#     device = 'mps'

if torch.cuda.is_available():
    device = 'cuda'

print(device)

def load_train_state(file_path, gen, disc):
    state = torch.load(file_path, weights_only=False)
    gen.load_state_dict(state['gen_model_state_dict'])
    disc.load_state_dict(state['disc_model_state_dict'])

    d_opt = torch.optim.RMSprop(disc.parameters(), lr=0.0001)
    g_opt = torch.optim.RMSprop(gen.parameters(), lr=0.0001)
    
    g_opt.load_state_dict(state['gen_optimizer_state_dict'])
    d_opt.load_state_dict(state['disc_optimizer_state_dict'])
    avg_dis_loss = state['disc_loss']
    avg_gen_loss = state['gen_loss']
    epoch = state['epoch']
    return (epoch, g_opt, d_opt, avg_gen_loss, avg_dis_loss)


def train(generator, discriminator, criterion, genr_optimizer, disc_optimizer, start_epoch=1, num_epochs=40, avg_gen_loss = None, avg_dis_loss = None):
    

    train_dataset = datasets.MNIST(root = "../data/",
                                   train = True,
                                   transform = transforms.ToTensor())

    loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    num_classes = len(train_dataset.classes)

    def train_step(model, x, target, criterion, optimizer, clip_threshold=None):
        model.train()
        pred = model(x)
        pred = pred.squeeze(dim=0)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if clip_threshold != None:
            with torch.no_grad():
                for param in model.parameters():
                    param.data.clamp_(-clip_threshold, clip_threshold)

        return loss.item()

    bestest = float('inf')
    if avg_gen_loss is not None:
        bestest = np.min(avg_gen_loss)
    else:
        avg_gen_loss = []
        avg_dis_loss = []
    
    progress = tqdm(range(start_epoch, start_epoch + num_epochs))

    for epoch in progress:
        dis_losses = []
        gen_losses = []
        fake_losses = []
        real_losses = []

        start_time = time.time()

        for step, (x, label) in enumerate(loader):
            batch_size = x.shape[0]
            real = torch.ones(batch_size, 1).to(device)
            fake = -torch.ones(batch_size, 1).to(device)

            x = x.to(device)
            real_loss = train_step(discriminator, x, real, criterion, disc_optimizer, 0.05)

            seed = createOnehotSeed(torch.ones(int(batch_size)), num_classes).to(device)
            x = generator(seed).detach()
            fake_loss = train_step(discriminator, x, fake, criterion, disc_optimizer, 0.05)

            fake_losses.append(fake_loss)
            real_losses.append(real_loss)
            
            if step != 0 and step % 5 == 0:
                seed = createOnehotSeed(label.reshape(-1), num_classes).to(device)
                x = generator(seed)
                gen_loss = train_step(discriminator, x, real, criterion, genr_optimizer)
                gen_losses.append(gen_loss)

            critic_loss = (real_loss + fake_loss) * 0.5
            dis_losses.append(critic_loss)

            progress.set_postfix_str(f"Epoch {epoch}, {step + 1}/{len(loader)}, dis_loss: {np.mean(dis_losses):.04f}, gen_loss: {np.mean(gen_losses):.04f}")
            # if step % 1000 == 0:
            #     show_plt(generator, num_classes, f'log/checkpoint_{epoch}_{step}.png')
            #     logging.info(f'---- Step {step}, DiscLoss: {np.mean(dis_losses):.04f}, GenLoss: {np.mean(gen_losses):.04f}')

        elapsed = time.time() - start_time
        avg_gen_loss = np.mean(gen_losses)
        avg_dis_loss = np.mean(dis_losses)

        show_plt(generator, num_classes, f'log/checkpoint_{epoch}_last.png')
        logging.info(f'Epoch {epoch}, elapsed: {elapsed}, DiscLoss: {avg_dis_loss:.04f}, real:{np.mean(real_losses):.04f}, fake:{np.mean(fake_losses):.04f}, GenLoss: {avg_gen_loss:.04f}')

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

def new_train():
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion = WassersteinLoss()

    disc_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)
    genr_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
    
    start_epoch = 1
    
    train(generator, discriminator, criterion, genr_optimizer, disc_optimizer, num_epochs=60)

def resume_train():
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion = WassersteinLoss()
    
    epoch, genr_optimizer, disc_optimizer, gen_losses, dis_losses = load_train_state("log/latest.pth", generator, discriminator)
    
    train(generator, discriminator, criterion, genr_optimizer, disc_optimizer, start_epoch=epoch+1, num_epochs=40, avg_gen_loss = gen_losses, avg_dis_loss = dis_losses)
    

if __name__ == "__main__":
    # train()

    # lr = 0.0001
    resume_train()