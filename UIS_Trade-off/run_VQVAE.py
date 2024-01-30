from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
import umap
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import argparse
import copy
import math
import os

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity as ssim

from PIL import Image
# from no_classifier_modules import Model
# from modules1_no_channel import Model
from modules_mod import Model
from mi import TMI, Perceptual
from modulation import QAM, PSK

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
# parser.add_argument('--intermediate_dim', type=int, default=128)
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--snr', type=int, default=15)
# parser.add_argument('--gamma', type=float, default=0.5)
# parser.add_argument('--decay_step', type=int, default=60)
# parser.add_argument('--beta', type=float, default=1e-2)
# parser.add_argument('--threshold', type=float, default=1e-2)
# args = parser.parse_args()

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
# parser.add_argument('--num_hiddens', type=int, default=128)
# parser.add_argument('--num_residual_hiddens', type=int, default=32)
# parser.add_argument('--num_residual_layers', type=int, default=5)
# parser.add_argument('--embedding_dim', type=int, default=64)
# parser.add_argument('--num_embeddings', type=int, default=512)
# parser.add_argument('--commitment_cost', type=float, default=0.25)
# parser.add_argument('--decay', type=float, default=-0.99)
# parser.add_argument('--learning_rate', type=float, default=0.001)
# parser.add_argument('--intermediate_dim', type=int, default=128)
# parser.add_argument('--snr', type=int, default=15)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--gamma', type=float, default=0.5)
# parser.add_argument('--decay_step', type=int, default=60)
# parser.add_argument('--beta', type=float, default=1e-2)
# parser.add_argument('--threshold', type=float, default=1e-2)
# args = parser.parse_args()

num_residual_hiddens = 32
num_residual_layers = 4
embedding_dim = 512
num_embeddings = 16
commitment_cost = 0.25

learning_rate = 1e-3
hidden_channel = 256

# snr = 15
epochs = 200
batch_size = 512

gamma = 0.5
decay_step = 60
beta = 1e-2
threshold = 1e-1
snr = 0.0

lambda_U = 1
lambda_I = 1e-2
lambda_S = 0.5

# Dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform_train)
test = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform_test)
# print(len(train))
# print(len(test))
dataset_train1_spl, dataset_train2_spl, _ = torch.utils.data.random_split(train, [25000, 25000, len(train) - 50000])
dataset_test_spl, _ = torch.utils.data.random_split(test, [10000, len(test) - 10000])
# dataset_test_spl = test
test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1000, shuffle=False, num_workers=2)  # 用于测试,batch_size原值为1000

# # SNR
# def SNR_to_noise(snr):
#     snr = 10 ** (snr / 10)
#     noise_std = 1 / np.sqrt(2 * snr)
#     return noise_std

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

# psnr计算
def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0**2 / mse)

def train_Decoder(model, Decoder_optimizer, dec_criterion, images, mod):
    model.Decoder.train()

    z = model.Encoder(images)
    z0 = z[0]
    z1 = z[1]
    z0 = PowerNormalize(z0)
    z = [z0, z1]
    vq_loss, input2_KL, quantized, encodings = model.vq_vae(z, mod=mod)
    x_hat = model.Decoder(quantized)

    loss = dec_criterion(x_hat, images) + vq_loss
    Decoder_optimizer.zero_grad()
    loss.backward()
    Decoder_optimizer.step()

    return loss

def train_trx(model, tx_optimizer, cx_optimizer, criterion, dec_criterion, images, target, epoch, mod):
    model.Encoder.train()
    model.Classifier.train()

    z = model.Encoder(images)
    z0 = z[0]
    z1 = z[1]
    z0 = PowerNormalize(z0)
    z = [z0, z1]
    vq_loss, input2_KL, quantized, encodings = model.vq_vae(z, mod=mod)
    quantized1 = quantized.contiguous()
    quantized1 = torch.reshape(quantized1, (quantized1.size()[0], 512 * 4 * 4))  # reshape输出为torch.Size([128, 128])
    y_hat, KL = model.Classifier(quantized1, input2_KL)
    x_hat = model.Decoder(quantized)

    loss1 = criterion(y_hat, target)
    if epoch <= 20:
        loss = loss1 + vq_loss
    else:
        anneal_ratio = min(1, (epoch - 20) / 20)
        loss = lambda_U * loss1 + lambda_I * KL * anneal_ratio + vq_loss
    if torch.isnan(loss):
        raise Exception("NaN value")

    dec_loss = dec_criterion(x_hat, images)
    loss = loss - lambda_S * dec_loss
    runningloss = loss.item()

    pred = y_hat.argmax(dim=1, keepdim=True)
    accuracy = pred.eq(target.view_as(pred))

    tx_optimizer.zero_grad()
    cx_optimizer.zero_grad()
    loss.backward()
    tx_optimizer.step()
    cx_optimizer.step()

    return accuracy, runningloss

def test_trx(model, test_data_loader, mod):
    model.Encoder.eval()
    model.Classifier.eval()
    correct = 0
    with torch.no_grad():
        for images, target in test_data_loader:
            # time_start = time.time()
            images = images.to(device)
            target = target.to(device)

            z = model.Encoder(images)
            z0 = z[0]
            z1 = z[1]
            z0 = PowerNormalize(z0)
            z = [z0, z1]
            vq_loss, input2_KL, quantized, encodings = model.vq_vae(z, mod=mod)
            quantized1 = quantized.contiguous()
            quantized1 = torch.reshape(quantized1, (quantized1.size()[0], 512 * 4 * 4))  # reshape输出为torch.Size([128, 128])
            y_hat, KL = model.Classifier(quantized1, input2_KL)

            # z, KL = model.Encoder(images)
            # z = PowerNormalize(z)
            # # channel_noise = SNR_to_noise(args.snr)
            # # z_hat = z + torch.normal(0, channel_noise, size=z.shape).to(device)
            # y_hat, _ = model.Classifier(z_hat, KL)
            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # time_end = time.time()
            # print('时间:', time_end - time_start)

        return correct / len(test_data_loader.dataset)
# 面向任务通信网络训练
def main_train():
    print('面向任务通信网络训练')
    test_acc = 0
    kwargs = {'num_workers': 1, 'pin_memory': True}
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    dec_criterion = nn.MSELoss()
    dec_criterion = dec_criterion.to(device)
    mod = QAM(num_embeddings, snr)
    model = Model(num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=mod).to(device)
    # model
    Encoder_optimizer = optim.Adam(model.Encoder.parameters(), lr=learning_rate, weight_decay=0.0001)
    Classifier_optimizer = optim.Adam(model.Classifier.parameters(), lr=learning_rate, weight_decay=0.0001)
    Decoder_optimizer = optim.Adam(model.Decoder.parameters(), lr=learning_rate, weight_decay=0.0001)
    Encoder_scheduler = StepLR(Encoder_optimizer, step_size=decay_step, gamma=gamma)
    Classifier_scheduler = StepLR(Classifier_optimizer, step_size=decay_step, gamma=gamma)
    Decoder_scheduler = StepLR(Decoder_optimizer, step_size=decay_step, gamma=gamma)
    # initNetParams(model)
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        print('This is {}-th epoch'.format(epoch))
        print('模拟对手')
        train2_data_loader = torch.utils.data.DataLoader(dataset_train2_spl, batch_size=batch_size, shuffle=True, **kwargs)
        for n, (images, _) in enumerate(train2_data_loader):
            images = images.to(device)
            for name, param in model.named_parameters():  # 冻结模型参数与更新的参数一块使用
                if "Encoder" in name:
                    param.requires_grad = False
                if "Classifier" in name:
                    param.requires_grad = False
                if "Decoder" in name:
                    param.requires_grad = True
            train_Decoder(model, Decoder_optimizer, dec_criterion, images, mod=mod)
        Decoder_scheduler.step()

        print('隐私通信')
        train1_data_loader = torch.utils.data.DataLoader(dataset_train1_spl, batch_size=batch_size, shuffle=True, **kwargs)
        total_correct = 0
        total_runningloss = 0
        for n, (images, target) in enumerate(train1_data_loader):
            images = images.to(device)
            target = target.to(device)
            for name, param in model.named_parameters(): # 冻结模型参数与更新的参数一块使用
                if "Encoder" in name:
                    param.requires_grad = True
                if "Classifier" in name:
                    param.requires_grad = True
                if "Decoder" in name:
                    param.requires_grad = False
            correct, runningloss = train_trx(model, Encoder_optimizer, Classifier_optimizer, criterion, dec_criterion, images, target, epoch, mod=mod)
            total_correct += correct.sum().item()
            total_runningloss += runningloss
        Encoder_scheduler.step()
        Classifier_scheduler.step()
        LOSS.append(total_runningloss / len(train1_data_loader.dataset))
        Loss0 = np.array(LOSS)
        np.save('./Loss/epoch_{}'.format(epoch), Loss0)
        print("训练损失：", total_runningloss / len(train1_data_loader.dataset))
        print("训练精度：", total_correct / len(train1_data_loader.dataset))

        acc = test_trx(model, test_data_loader, mod)
        print('测试精度:', acc)
        ACC.append(acc)
        testacc = np.array(ACC)
        np.save('./TestAcc/epoch_{}'.format(epoch),testacc)
        if acc > test_acc:
            test_acc = acc
            saved_model = copy.deepcopy(model.state_dict())
            with open('./model_mod/CIFAR_SNR_{}_epoch_{}.pth'.format(snr, epoch),'wb') as f:
                torch.save({'model': saved_model}, f)

# 测试分类精度
def main_test(): # 测试分类精度
    mod = QAM(num_embeddings, snr)
    model = Model(num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=mod).to(device)
    model.load_state_dict(torch.load('./model_mod/CIFAR_SNR_12.0_epoch_193.pth')['model'])
    accuracy = 0
    t = 5
    for i in range(t):
        acc = test_trx(model, test_data_loader, mod)
        accuracy += acc
    print('测试精度:', accuracy / t)

# 模型反演攻击
def main_train_dec(): # 训练模型反演攻击时的信噪比
    kwargs = {'num_workers': 2, 'pin_memory': True}
    criterion1 = nn.MSELoss()
    criterion1 = criterion1.to(device)
    per = Perceptual()
    mod = QAM(num_embeddings, snr)
    tmi = TMI(num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=mod).to(device)
    optimizer = optim.Adam(tmi.parameters(), lr=learning_rate, weight_decay=0.0001)
    print('模型反演攻击')
    for epoch in range(epochs):
        print('This is {}-th epoch'.format(epoch))
        train2_data_loader = torch.utils.data.DataLoader(dataset_train2_spl, batch_size=batch_size, shuffle=True, **kwargs)
        for n, (images, _) in enumerate(train2_data_loader):
            images = images.to(device)
            x_hat, vq_loss = tmi(images, mod=mod)
            # loss = criterion1(x_hat, images) + per(x_hat, images) + vq_loss
            loss = criterion1(x_hat, images) + per(x_hat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    saved_model = copy.deepcopy(tmi.state_dict())
    with open('./model_mod/MI_SNR_{}_epoch_{}.pth'.format(snr, epoch), 'wb') as f:
        torch.save({'tmi': saved_model}, f)

# 测试模型反演攻击
def main_test_dec():
    mod = QAM(num_embeddings, snr)
    tmi = TMI(num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, hidden_channel, snr, mod=mod).to(device)
    tmi.load_state_dict(torch.load('./model_mod/MI_SNR_12.0_epoch_199.pth')['tmi'])
    test_data_loader = torch.utils.data.DataLoader(dataset_test_spl, batch_size=1, shuffle=False, num_workers=1)  # 用于测试
    path1 = './fake_image/sampled-'
    path2 = './true_image/sampled-'
    list_psnr = []
    tmi.eval()
    with torch.no_grad():
        for n, (images, _) in enumerate(test_data_loader):
            images = images.to(device)
            x_hat, vq_loss = tmi(images, mod=mod)
            save_image(x_hat, os.path.join('fake_image', 'sampled-{}.png'.format(n)))
            save_image(images, os.path.join('true_image', 'sampled-{}.png'.format(n)))
            img_a = Image.open(path1 + str(n) + '.png')
            img_b = Image.open(path2 + str(n) + '.png')
            img_a = np.array(img_a)
            img_b = np.array(img_b)
            # win_size = 3
            # ssim_value = ssim(img_a, img_b, win_size=win_size, multichannel=True) # win_size = win_size
            # list_psnr.append(ssim_value)
            psnr_num = psnr(img_a, img_b)
            list_psnr.append(psnr_num)
        print(np.mean(list_psnr))

if __name__ == '__main__':
    # seed_torch(0)
    # main_train() # 训练任务通信网络
    main_test()  # 测试任务通信网络
    # main_train_dec() # 训练模型反演攻击网络
    # main_test_dec() # 训练模型反演攻击网络







