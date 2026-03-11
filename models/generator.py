import torch.nn as nn

class Generator(nn.Module):

    def __init__(self,z_dim=128):

        super().__init__()

        self.fc = nn.Linear(z_dim,256*8*10*10)

        self.net = nn.Sequential(

            nn.ConvTranspose3d(256,128,4,2,1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128,64,4,2,1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64,32,4,2,1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32,4,4,2,1),

            nn.Tanh()
        )

    def forward(self,z):

        x = self.fc(z)

        x = x.view(-1,256,8,10,10)

        return self.net(x)