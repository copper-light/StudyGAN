import torch
from torch import nn

import matplotlib.pyplot as plt

device = 'cpu'
# if torch.mps.is_available():
#     device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None, batch_norm=True, dropout=0.):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(activation)

        if dropout > 0.:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, out_padding, activation=None, norm=True, dropout=0.):
        super().__init__()
        layers = []
        
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding, bias=not norm)
        layers.append(self.convT)

        if norm:
            self.norm = True
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(activation())

        if dropout > 0.:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)


    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append(ConvBlock(1, 32, 3, 1, 1, nn.LeakyReLU(0.2), True, 0.2)) # 28
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(ConvBlock(32, 64, 3, 1, 1, nn.LeakyReLU(0.2), True, 0.2)) # 14
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(ConvBlock(64, 8, 3, 1, 1, nn.LeakyReLU(0.2))) # 7
        self.featureBlock = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(8 * 7 * 7, 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.featureBlock(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        block = TransposeConvBlock(1, 32, 3, 1, 0, 0, nn.ReLU, True, 0.2)  # 12
        layers.append(block)
        block = TransposeConvBlock(32, 64, 3, 2, 0, 1, nn.ReLU, True, 0.2)  # 24
        layers.append(block)
        block = TransposeConvBlock(64, 1, 3, 1, 0, 0, nn.ReLU, True)  # 28
        layers.append(block)

        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequence(x)
        return x


def createOnehot2DMatrix(class_indexes, classes_num):
    onehot = torch.zeros([len(class_indexes), 1, classes_num, classes_num])
    for i, value in enumerate(class_indexes):
        onehot[i, 0, int(value), int(value)] = 1.
    return onehot


def createOnehot2DSeed(class_indexes, classes_num):
    # onehot = createOnehot2DMatrix(class_indexes, classes_num)
    seed = torch.randn(len(class_indexes), 1, 10, 10)
    # seed = torch.concat([seed, onehot], dim=1)
    return seed

def show_plt(generator, num_of_classes, save_path = None):
    fig, axes = plt.subplots(1, num_of_classes, figsize=(15, 6))  # 2행 5열 격자 생성

    classes = [v for v in range(num_of_classes)]

    seed = createOnehot2DSeed(classes, num_of_classes)
    seed = seed.to(device)
    images = generator(seed)

    for i in range(num_of_classes):
        ax = axes[i]
        image = images[i].reshape(28, 28).cpu()

        # 예시로 각 그림에 숫자 표시
        ax.imshow(image.detach().numpy(), cmap='gray')
        ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    block1 = TransposeConvBlock(1, 32, 3, 1, 0, 0, nn.ReLU, True) # 12
    block2 = TransposeConvBlock(32, 64, 3, 2, 0, 1, nn.ReLU, True) # 26
    block3 = TransposeConvBlock(64, 1, 3, 1, 0, 0, nn.ReLU, True) # 28
    seed = createOnehot2DSeed([2], 10)
    # print(seed.shape)

    out = block1(seed)
    print(out.shape)
    out = block2(out)
    print(out.shape)
    out = block3(out)
    # out = block4(out)
    # out = block5(out)
    print(out.shape)

    model = Generator()
    images = model(seed)
    print(images.shape)

    model = Discriminator()
    pred = model(images)
    pred = torch.squeeze(pred, dim=0)
    print(pred)
    
    # gen = Generator()
    # seed = torch.rand(4, 100)
    # one_hot = nn.functional.one_hot(torch.tensor([1,2,3,4]), 10).type(torch.FloatTensor)
    # seed = torch.concat([seed, one_hot], dim=1)
    # print(seed[0])
    # print(seed.shape)
    # pred = gen(seed)
    # print(pred.shape)
