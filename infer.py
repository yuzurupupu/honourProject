import torch
from models.generator import Generator


G = Generator(128).cuda()

G.load_state_dict(
    torch.load("checkpoints/model_200.pth")["models"]["G"]
)

z = torch.randn(1,128).cuda()

fake = G(z)