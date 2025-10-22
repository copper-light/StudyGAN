import matplotlib.pyplot as plt
import torch

device = 'cpu'
if torch.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'

def show_plt(generator, num_of_classes):
    fig, axes = plt.subplots(1, 10, figsize=(15, 6))  # 2행 5열 격자 생성

    for i in range(10):
        ax = axes[i]

        seed = torch.randn(100).to(device)
        # one_hot = torch.nn.functional.one_hot(torch.tensor(i), num_of_classes).to(device)
        # seed = torch.concat([seed, one_hot], dim=0).to(device)
        image = generator(seed)
        image = image.reshape(28, 28).cpu()

        # 예시로 각 그림에 숫자 표시
        ax.imshow(image.detach().numpy(), cmap='gray')
        ax.axis('off')  # 축 숨기기

    plt.tight_layout()
    plt.show()
