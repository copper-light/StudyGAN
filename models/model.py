import torch.nn as nn
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, input_dim, output_dim, name="GAN", num_classes = 0, device = 'cpu', is_train = True, lr=1e-4):
        self.device = device
        self.name = name
        self.num_classes = num_classes
        self.is_train = is_train
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.G = None
        self.D = None
        self.G_optimizer = None
        self.D_optimizer = None

        self._setup_model( input_dim, output_dim, num_classes, device, is_train, lr)

    @abstractmethod
    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        pass

    @abstractmethod
    def train_discriminator(self, x, target):
        pass

    @abstractmethod
    def train_generator(self, x, target):
        pass

    @abstractmethod
    def _generate_seed(self, labels):
        pass

    @abstractmethod
    def generate_image_to_numpy(self, size):
        pass

    @abstractmethod
    def get_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass