import torch.nn as nn
from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def train_discriminator(self, x, y):
        pass

    @abstractmethod
    def train_generator(self, x, y):
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