import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList([
            vgg[:4],
            vgg[:9],
            vgg[:16]
        ])

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        )

    def forward(self, fake, real, mask):

        fake = (fake + 1) / 2
        real = (real + 1) / 2

        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std

        loss = 0
        for layer in self.layers:
            f_fake = layer(fake)
            f_real = layer(real)
            loss += torch.mean(torch.abs(f_fake - f_real))

        return loss

