import torch.nn as nn

class CodeDiscriminator(nn.Module):

    def __init__(self,z_dim=128):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.2),

            nn.Linear(256,256),
            nn.LeakyReLU(0.2),

            nn.Linear(256,1)
        )

    def forward(self,z):

        return self.net(z)