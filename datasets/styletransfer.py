import os
import numpy as np
import h5py

from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class StyleTransferDataset(Dataset):

    def __init__(self, root, limit = None, cache = True, train = True, transform=None, shuffle=True):
        super(StyleTransferDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.cache = cache
        self.limit = limit
        self.cachefile = None
        self.shuffle = shuffle

        np.random.seed(42)
        
        middle_path = 'train'
        if not train:
            middle_path = 'test'

        if cache:
            cache_file_path = os.path.join(root, f'{middle_path}.h5')
            if not os.path.exists(cache_file_path):
                a_path = glob(os.path.join(self.root, f'{middle_path}A/*.jpg'))
                b_path = glob(os.path.join(self.root, f'{middle_path}B/*.jpg'))
                self.a = np.array([self._open_image(p) for p in a_path]).astype(np.float32)
                self.b = np.array([self._open_image(p) for p in b_path]).astype(np.float32)
                with h5py.File(cache_file_path, 'w') as f:
                    f.create_dataset('trainA', data=self.a)
                    f.create_dataset('trainB', data=self.b)

            self.cachefile = h5py.File(cache_file_path, 'r')
            self.a = self.cachefile['trainA'][:]
            self.b = self.cachefile['trainB'][:]
        else:
            self.a = glob(os.path.join(self.root, f'{middle_path}A/*.jpg'))
            self.b = glob(os.path.join(self.root, f'{middle_path}B/*.jpg'))

        if self.limit is not None:
            self.a = self.a[:self.limit]
            self.b = self.b[:self.limit]

        self.len = min(len(self.a), len(self.b))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        a = None
        b = None
        if self.cache:
            a = self.a[index]
            b = self.b[index]
        else:
            a = np.array(self._open_image(self.a[index])).astype(np.float32) / 255.
            b = np.array(self._open_image(self.b[index])).astype(np.float32) / 255.

        if self.transform:
            return self.transform(a).float(), self.transform(b).float()
        else:
            return a, b

    def _open_image(self, path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def on_epoch_start(self):
        if self.shuffle:
            np.random.shuffle(self.a)
            np.random.shuffle(self.b)

if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader

    dataset = StyleTransferDataset("../data/horse2zebra", train=True, cache=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("train_dataset", len(dataset))
    for a, b in dataset:
        print(a, b)
        break

    cnt = 0
    for a, b in loader:
        cnt += 1


