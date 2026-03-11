import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            spectral_norm(nn.Conv3d(4,32,4,2,1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv3d(32,64,4,2,1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv3d(64,128,4,2,1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv3d(128,256,4,2,1)),
            nn.LeakyReLU(0.2)

        )

        self.fc = nn.Linear(256*8*10*10,1)

    def forward(self,x):

        x = self.net(x)

        x = x.view(x.size(0),-1)

        return self.fc(x)