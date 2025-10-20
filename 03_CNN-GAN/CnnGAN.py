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

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation:
            layers.append(activation())

        if dropout > 0.:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride, padding, out_padding, activation=None, norm=True, dropout=0.):
        super().__init__()
        layers = []
        
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding, bias=True)
        layers.append(self.convT)

        if norm:
            self.norm = True
            layers.append(nn.LayerNorm(in_channels, input_size, input_size))

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

        self.conv1 = ConvBlock(1, 8, 3, 1, 1) # 28
        # layers.append(nn.functional.max_pool2d(2)) # 14
        self.conv2 = ConvBlock(8, 16, 3, 1, 1) # 14
        # layers.append(nn.functional.max_pool2d(2))  # 7
        self.conv3 = ConvBlock(16, 32, 3, 1, 1) # 7
        # self.featureBlock = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(32 * 7 * 7, 1024))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(1024, 512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(512, 1))
        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        block = TransposeConvBlock(10, 1, 4, 5, 1, 0, 0, nn.ReLU, True)  # 14
        layers.append(block)
        block = TransposeConvBlock(14, 4, 8, 5, 1, 0, 0, nn.ReLU, True)  # 18
        layers.append(block)
        block = TransposeConvBlock(18, 8, 16, 5, 1, 0, 0, nn.ReLU, True)  # 22
        layers.append(block)
        block = TransposeConvBlock(22, 16, 16, 5, 1, 0, 0, nn.ReLU, True)  # 26
        layers.append(block)
        block = TransposeConvBlock(26, 1, 5, 1, 1, 0, nn.Sigmoid, True)  # 28
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

def show_plt(generator, num_of_classes):
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
    plt.show()


if __name__ == "__main__":
    block1 = TransposeConvBlock(2, 4, 5, 1, 0, 0) # 14
    block2 = TransposeConvBlock(4, 8, 5, 1, 0, 0) # 18
    block3 = TransposeConvBlock(8, 16, 5, 1, 0, 0) # 22
    block4 = TransposeConvBlock(16, 16, 5, 1, 0, 0) # 26
    block5 = TransposeConvBlock(16, 1, 5, 1, 1, 0)  # 28
    seed = createOnehot2DSeed([2], 10)
    print(seed.shape)

    out = block1(seed)
    out = block2(out)
    out = block3(out)
    out = block4(out)
    out = block5(out)
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
