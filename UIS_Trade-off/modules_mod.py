import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):  # 128 128 32
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
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, hidden_channel, snr):
        super(Encoder, self).__init__()
        self.hidden_channel = hidden_channel
        self.snr = snr
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer1_res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.encoder0 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self._residual_stack = ResidualStack(in_channels=128, num_hiddens=128, num_residual_layers=1,
                                             num_residual_hiddens=32)
        self.encoder2 = nn.Sequential(
            nn.Linear(8192, 256),
            nn.Sigmoid()  # 正式得还回去
        )

        self.encoder3_weight = nn.Parameter(torch.Tensor(self.hidden_channel, 256))
        self.encoder3_bias = nn.Parameter(torch.Tensor(self.hidden_channel))
        self.encoder3_weight.data.normal_(0, 0.5)
        self.encoder3_bias.data.normal_(0, 0.1)
        self.upper_tri_matrix = torch.triu(torch.ones((self.hidden_channel, self.hidden_channel))).to(device)

    def forward(self, x0):
        x = self.prep(x0)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self._residual_stack(x)
        x1 = x.contiguous()
        x1 = torch.reshape(x1, (x1.size()[0], 512 * 4 * 4))

        x1 = self.encoder2(x1)
        x1_norm2 = torch.norm(x1, dim=1)
        x1 = 256 * (x1.permute(1, 0) / (x1_norm2 + 1e-6)).permute(1, 0)
        # print(x.shape)
        x1 = x1.to(device)
        weight3 = F.tanh(self.encoder3_weight)
        bias3 = F.tanh(self.encoder3_bias)
        weight3 = torch.clamp(torch.abs(weight3), min=1e-3) * torch.sign(weight3.detach())
        bias3 = torch.clamp(torch.abs(bias3), min=1e-3) * torch.sign(bias3.detach())
        l2_norm_squared = torch.sum(weight3.pow(2), dim=1) + bias3.pow(2)
        l2_norm = l2_norm_squared.pow(0.5)
        weight3 = (weight3.permute(1, 0) / (l2_norm + 1e-6)).permute(1, 0)
        weight3 = weight3.to(device)
        bias3 = bias3 / (l2_norm + 1e-6)
        bias3 = bias3.to(device)
        x1 = F.linear(x1, weight3, bias3)

        mu = nn.Parameter(torch.ones(self.hidden_channel)).to(device)
        mu = F.linear(mu, self.upper_tri_matrix)
        mu = torch.clamp(mu, min=1e-4).to(device)  # torch.Size([128])
        # print(mu.shape)

        encoded_feature = torch.tanh(x1 * mu)  # torch.Size([128, 128])
        # print(encoded_feature.shape)
        encoded_feature = torch.clamp(torch.abs(encoded_feature), min=1e-2) * torch.sign(
            encoded_feature.detach())  # torch.Size([128, 128])
        # print(encoded_feature.shape)

        noise = self.SNR_to_noise(self.snr)
        channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)  # torch.Size([1])
        # print(channel_noise.shape)

        # KL divergence
        KL = self.KL_log_uniform(channel_noise, torch.abs(encoded_feature))
        # print(KL.shape)
        return x, KL

    def KL_log_uniform(self, channel_noise, encoded_feature):
        alpha = (channel_noise / encoded_feature)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        return - torch.sum(KL_term) / batch_size

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim  # embedding_dim = 512
        self._num_embeddings = num_embeddings  # num_embeddings = 512 改成 16

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)  # 16X64
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def mod_channel_demod(self, mod, x, device):
        X = mod.modulate(x)
        X = mod.awgn(X)
        # return mod.demodulate(X).to(self._embedding.device)
        return mod.demodulate(X).to(device)

    def construct_noise(self, encodings, mod, device):
        x = torch.argmax(encodings, dim=-1)
        x_tilde = self.mod_channel_demod(mod, x, device)
        noise = F.one_hot(x_tilde, num_classes=self._num_embeddings).float() - \
                F.one_hot(x, num_classes=self._num_embeddings).float()
        return noise

    def recover(self, encodings):
        out = torch.matmul(encodings, self._embedding.weight)
        return out

    def forward(self, inputs, mod):
        # convert inputs from BCHW -> BHWC
        input1 = inputs[0]    # torch.Size([512, 512, 4 ,4])
        input2_KL = inputs[1]
        # print(input1.shape)
        input1 = input1.permute(0, 2, 3, 1).contiguous()   # torch.Size([512, 4, 4, 512])
        # print(type(input1))
        input_shape = input1.shape

        # Flatten input
        flat_input = input1.view(-1, self._embedding_dim)  # torch.Size([8192, 512])

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))  # torch.Size([8192, 16])
        # print(distances.shape)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # torch.Size([8192, 1])
        # print(encoding_indices.shape)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=input1.device)  # torch.Size([8192, 16])
        # print(encodings.shape)
        encodings.scatter_(1, encoding_indices, 1)  # torch.Size([8192, 16])
        # print(encodings.shape)

        # 调制与解调
        noise = self.construct_noise(encodings, mod=mod, device=input1.device)
        encodings = encodings + noise
        quantized = self.recover(encodings)
        quantized = quantized.view(input_shape)
        # Quantize and unflatten
        # quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)  # torch.Size([256, 8, 8, 64])
        # print(quantized.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), input1)
        q_latent_loss = F.mse_loss(quantized, input1.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = input1 + (quantized - input1).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        # return loss, input2_KL, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, input2_KL, quantized.permute(0, 3, 1, 2).contiguous(), encodings
"""
源代码的Decoder
"""
class Decoder(nn.Module):
    def __init__(self, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self.decov1 = nn.Sequential(nn.Conv2d(512, 256,kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.decov2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.decov3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self._residual_stack = ResidualStack(in_channels=128, num_hiddens=128, num_residual_layers = 2, num_residual_hiddens = num_residual_hiddens)
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

class Classifier(nn.Module):
    def __init__(self, snr):
        super(Classifier, self).__init__()
        self.snr = snr
        self.decoder1 = nn.Linear(8192, 8192)
        self.decoder1_2 = nn.Sequential(
            nn.Linear(8192, 8192),
            nn.ReLU()
        )
        self.decoder1_2_2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.decoder1_3 = nn.Sequential(
            nn.Linear(8192 + 16, 8192),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.classifier1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            Flatten()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512, 10, bias=False),
            Mul(0.125)
        )

    def forward(self, x, KL):
        noise = self.SNR_to_noise(self.snr)
        channel_noise = torch.FloatTensor([1]) * noise
        channel_noise = channel_noise.to(device)
        x = F.relu(self.decoder1(x))
        x = self.decoder1_2(x)
        noise_feature = self.decoder1_2_2(channel_noise)
        noise_feature = noise_feature.expand(x.size()[0], 16)
        x = torch.cat((x, noise_feature), dim=1)
        x = self.decoder1_3(x)
        x = torch.reshape(x, (-1, 512, 4, 4))
        decoded_feature = self.decoder2(x)    # torch.Size([512, 512, 4, 4])
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature  # torch.Size([512, 512, 4, 4])
        x = self.classifier1(x)  # torch.Size([512, 512])
        output = self.classifier2(x)  # torch.Size([128, 10])

        return output, KL * 0.1 / channel_noise

    def SNR_to_noise(self, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        return noise_std

class Model(nn.Module):
    def __init__(self, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=None):
        super(Model, self).__init__()
        self.Encoder = Encoder(hidden_channel, snr)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.Classifier = Classifier(snr)
        self.Decoder = Decoder(num_residual_layers, num_residual_hiddens)

    def forward(self, x, mod=None):
        z = self.Encoder(x)
        loss, input2_KL, quantized, encodings = self.vq_vae(z, mod=mod)
        quantized1 = quantized.contiguous()
        quantized1 = torch.reshape(quantized1, (quantized1.size()[0], 512 * 4 * 4))  # reshape输出为torch.Size([128, 128])
        output, KL = self.Classifier(quantized1, input2_KL)
        x_recon = self.Decoder(quantized)

        return loss, x_recon, output, KL
