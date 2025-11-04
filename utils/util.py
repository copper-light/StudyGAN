import argparse
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def show_plt(images, n_rows=10, n_cols=10, show = False, save_path = None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    image_index = 0
    if n_rows == 1:
        for i in range(n_cols):
            ax = axes[i]
            image = images[image_index]  # numpy
            if images.shape[1] == 1:
                image = image.reshape(28, 28)
                ax.imshow(image, cmap='gray')
            else:
                image = np.transpose(image, (2, 1, 0))
                image = image * 255
                image = image.astype(np.uint8)
                ax.imshow(image)
            # image = image * 0.5 + 0.5
            image_index = image_index + 1

            ax.axis('off')  # 축 숨기기
    else:
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i,j]
                image = images[image_index] # numpy
                if images.shape[1] == 1:
                    image = image.reshape(28, 28)
                    ax.imshow(image, cmap='gray')
                else:
                    image = np.transpose(image, (2, 1, 0))
                    image = image * 255
                    image = image.astype(np.uint8)
                    ax.imshow(image)
                # image = image * 0.5 + 0.5
                image_index = image_index + 1

                ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf)
    image_array = np.array(im)
    buf.close()

    return image_array


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')