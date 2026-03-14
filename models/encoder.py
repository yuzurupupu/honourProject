import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim=1000, channel=512):
        super().__init__()
        self.conv1 = nn.Conv3d(1, channel // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel // 4)
        self.conv3 = nn.Conv3d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel // 2)
        self.conv4 = nn.Conv3d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, z_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        return h5.view(x.size(0), -1)