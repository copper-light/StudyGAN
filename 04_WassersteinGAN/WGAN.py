import torch
import torch.nn as nn

import matplotlib.pyplot as plt

device = 'cpu'
# if torch.mps.is_available():
#     device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        return -torch.mean(preds * targets)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 28
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # 14
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),

            # 7
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(100, 3136),
            nn.BatchNorm1d(3136),
            nn.ReLU()
        )

        self.feature = nn.Sequential(

            # 7 -> 14
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 14 -> 28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 28 -> 28
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid() # tanh 와 비교해볼 필요 있음 -> 이미지에서 tanh
        )

    def forward(self, x):
        x = self.input(x)
        x = x.view(x.size(0), 64, 7, 7)
        return self.feature(x)


def createOnehotSeed(class_indexes, classes_num):
    # onehot = createOnehot2DMatrix(class_indexes, classes_num)
    seed = torch.randn(len(class_indexes), 100)
    # seed = torch.concat([seed, onehot], dim=1)
    return seed


def show_plt(generator, num_of_classes, save_path = None):
    fig, axes = plt.subplots(1, num_of_classes, figsize=(15, 6))  # 2행 5열 격자 생성

    classes = [v for v in range(num_of_classes)]

    seed = createOnehotSeed(classes, num_of_classes)
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

    plt.show()


if __name__ == '__main__':

    gen = Generator()
    seed = torch.randn((64, 100))
    images = gen(seed)
    print(images.shape)

    dis = Discriminator()
    pred = dis(images)
    print(pred.shape)
