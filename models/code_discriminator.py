import torch.nn as nn


class CodeDiscriminator(nn.Module):
    def __init__(self, z_dim=1000, num_units=750):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, num_units),
            nn.LayerNorm(num_units),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.LayerNorm(num_units),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.l3 = nn.Linear(num_units, 1)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return h3