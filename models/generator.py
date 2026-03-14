import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, noise=1000, channel=64):
        super().__init__()
        c = channel
        self.noise = noise
        self.relu = nn.ReLU(inplace=True)

        self.tp_conv1 = nn.ConvTranspose3d(noise, c * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(c * 8)

        self.tp_conv2 = nn.Conv3d(c * 8, c * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(c * 4)

        self.tp_conv3 = nn.Conv3d(c * 4, c * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(c * 2)

        self.tp_conv4 = nn.Conv3d(c * 2, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(c)

        self.tp_conv5 = nn.Conv3d(c, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, noise):
        noise = noise.view(-1, self.noise, 1, 1, 1)

        h = self.tp_conv1(noise)
        h = self.relu(self.bn1(h))

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.tp_conv5(h)

        h = torch.tanh(h)
        return h