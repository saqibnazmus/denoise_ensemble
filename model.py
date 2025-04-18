import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
from skimage.metrics import structural_similarity as ssim

# Custom residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x + residual

# Multi-activation feature ensemble module
class ActC(nn.Module):
    def __init__(self, in_channels):
        super(ActC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 7, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 7, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, 7, kernel_size=7, padding=3)
        self.final_conv = nn.Conv2d(21, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(x)
        x2 = torch.sigmoid(x)
        x3 = x2 * x
        x4 = F.softplus(x)
        x4 = torch.tanh(x4)
        x5 = x4 * x
        c1 = self.conv1(x1)
        c2 = self.conv2(x3)
        c3 = self.conv3(x5)
        cx = torch.cat([c1, c2, c3], dim=1)
        return self.final_conv(cx)

# Residual feature aggregation module
class MDSR1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDSR1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.res_block1 = ResidualBlock(out_channels, out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        self.res_block3 = ResidualBlock(out_channels, out_channels)
        self.final_conv = nn.Conv2d(out_channels * 4, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.res_block1(x)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x1)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.final_conv(x)

# Multi-activation cascaded aggregation
class RCAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCAT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=7, padding=3)
        self.final_conv = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, padding=0)
        self.actc = ActC(16)

    def forward(self, x):
        y1 = self.conv1(x)
        x1 = self.conv2(x)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        a1 = self.actc(x1)
        a2 = self.actc(y1)
        c1 = torch.cat([a1, a2], dim=1)
        c2 = self.final_conv(c1)
        return x + c2

# Proposed model
class DenoisingModel(nn.Module):
    def __init__(self):
        super(DenoisingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.actc = ActC(64)
        self.mdsr1 = MDSR1(64, 32)
        self.rdn = ResidualBlock(64, 128)
        self.rcat = RCAT(64, 16)
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, dilation=2, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        f1 = self.actc(x)
        f2 = self.mdsr1(x, 32)
        f3 = self.rdn(x, 128)
        f4 = self.rcat(x, 16)
        inp = torch.cat([f1, f2, f3, f4], dim=1)
        return self.final_conv(inp)


