import matplotlib.pyplot as plt

def show_plt(images, n_rows=1, n_cols=10, show = False, save_path = None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for i in range(n_cols):
        ax = axes[i]
        image = images[i] # numpy
        if images.shape[1] == 1:
            image = images[i].reshape(28, 28)
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image, cmap='rgb')
        # image = image * 0.5 + 0.5

        # 예시로 각 그림에 숫자 표시

        ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()