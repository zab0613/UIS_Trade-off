import math
import numpy as np

import torch
import torch.nn as nn
from modules_mod import Model
from torchvision.models import vgg
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

# SNR
def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


# 模型反演攻击模块
class MI(nn.Module):
    def __init__(self, num_residual_layers, num_residual_hiddens):
        super(MI, self).__init__()
        self.decov1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.decov2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.decov3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self._residual_stack = ResidualStack(in_channels=128, num_hiddens=128, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self.decov4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.decov5 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(3), nn.Sigmoid())

    def forward(self, inputs):
        x = self.decov1(inputs)
        x = self.decov2(x)
        x = self.decov3(x)
        x = self._residual_stack(x)
        x = self.decov4(x)
        x = self.decov5(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        cnn = vgg.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:5]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), cnn[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)
        return loss

class TMI(nn.Module):
    def __init__(self, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=None):
        super(TMI, self).__init__()
        self.model = Model(num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=mod)
        self.model.load_state_dict(torch.load('model_mod/CIFAR_SNR_12.0_epoch_193.pth')['model'])
        self.mi = MI(num_residual_layers, num_residual_hiddens)

    def forward(self, x, mod=None):
        z = self.model.Encoder(x)
        z0 = z[0]
        z1 = z[1]
        z0 = PowerNormalize(z0)
        z = [z0, z1]
        loss, input2_KL, quantized, encodings = self.model.vq_vae(z, mod=mod)
        quantized = quantized.detach()
        x_hat = self.mi(quantized)

        return x_hat, loss