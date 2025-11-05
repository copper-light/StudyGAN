import argparse
import numpy as np
import matplotlib.pyplot as plt

def show_plt(images, n_rows=1, n_cols=10, show = False, save_path = None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for i in range(n_cols):
        ax = axes[i]
        image = images[i] # numpy
        image = image * 0.5 + 0.5
        if images.shape[1] == 1:
            image = images[i].reshape(28, 28)
            ax.imshow(image, cmap='gray')
        else:
            image = np.transpose(image, (2, 1, 0))

            image = image * 255
            image = image.astype(np.uint8)
            ax.imshow(image)

        ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')