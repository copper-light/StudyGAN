import torch
import torch.optim as optim
import torch.nn as nn
from typing import Any

from torchvision.models import vgg19, VGG19_Weights

from models.model import Model


class Encoder(nn.Module):
    def __init__(self, conv_indexes = (0,5,10,19,28)):
        super().__init__()
        model = vgg19(weights = VGG19_Weights.DEFAULT)
        model = model.features[:31]
        modules = []
        layers = []
        for i, l in enumerate(model.children()):
            layers.append(l)
            if i in conv_indexes:
                modules.append(nn.Sequential(*layers))
                layers = []
        if len(layers) > 0:
            modules.append(nn.Sequential(*layers))

        self.modules = modules

    def forward(self, x):
        middle_outputs = []
        for m in self.modules:
            x = m(x)
            middle_outputs.append(x)
        return x, middle_outputs


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class StyleLoss(nn.Module):
    def __init__(self, c,h,w):
        super().__init__()
        self.norm_value = 4.0 * (c ** 2) * ((h * w) ** 2)

    def _gram_matrix(self, x):
        # x shape must be (c,h,w)
        x = x.view(x.shape[0], -1)
        mat = torch.mm(x, torch.t(x))
        return mat

    def forward(self, pred, target):
        target = self._gram_matrix(target)
        pred = self._gram_matrix(pred)

        style_loss = torch.sum((target - pred) ** 2) / self.norm_value
        return style_loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        h = torch.square(image[:, :, :-1, :-1] - image[:, :, 1:, :-1])
        w = torch.square(image[:, :, :-1, :-1] - image[:, :, -1:, 1:])
        return torch.sum(torch.pow(h + w, 1.25))


class NeuralStyleTransfer(Model):

    def __init__(self, content_weight = 1, style_weight = 100, total_variation_weight = 20, **kwargs: Any):
        super().__init__(**kwargs)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight

    def _setup_model(self, input_dim, output_dim, num_classes, device, is_train, lr):
        self.model = Encoder()
        self.criterion_content = ContentLoss()
        self.criterion_style = StyleLoss(*input_dim)
        self.criterion_tv = TotalVariationLoss()
        self.combination_image = torch.randn(1, input_dim[0], input_dim[1], input_dim[2], device = device, requires_grad = True, dtype = torch.float)
        self.optimizer = optim.Adam([self.combination_image], lr=lr)

    def train_discriminator(self, x, target) -> tuple[float, dict]:
        return .0, None

    def train_generator(self, base, style) -> tuple[torch.Tensor, dict]:
        self.model.eval()
        self.optimizer.zero_grad()
        base = base.to(self.device)
        style = style.to(self.device)

        x = torch.concat([base, style, self.combination_image], dim=0)
        last_feature, middle_features = self.model(x)

        content_loss = self.criterion_content(last_feature[2], last_feature[0]) * self.content_weight

        style_loss = torch.tensor(.0)
        for feature in middle_features:
            style_loss += self.criterion_style(feature[2], feature[1])
        style_loss = (style_loss / len(middle_features)) * self.style_weight

        tv_loss = self.criterion_tv(self.combination_image) * self.total_variation_weight

        loss = content_loss + style_loss + tv_loss
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'content_loss': content_loss.item(), 'style_loss': style_loss.item(), 'tv_loss': tv_loss.item()}

    def _generate_seed(self, labels):
        return None

    def generate_image_to_numpy(self, x, y):
        return torch.concat([
            x,
            y,
            self.combination_image.detach().cpu(),
        ]).cpu().numpy()

    def get_checkpoint(self):
        pass

    def load_checkpoint(self, checkpoint):
        pass


if __name__ == "__main__":
    encoder = Encoder()

    base = torch.randn(1, 3, 224, 224)
    style = torch.randn(1, 3, 224, 224)
    NST = NeuralStyleTransfer(input_dim=(3, 224, 224), output_dim=(3, 224, 224))
    print(NST.combination_image[0, 0, 0, 0])
    loss, state = NST.train_generator(base, style)
    print(loss, state, NST.combination_image[0, 0, 0, 0])
    loss, state = NST.train_generator(base, style)
    print(loss, state, NST.combination_image[0, 0, 0, 0])