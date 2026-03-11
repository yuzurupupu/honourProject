import torch
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
from models.code_discriminator import CodeDiscriminator

from trainer import Trainer

from configs import *

dataset = BraTSDataset("processed_brats2021/train")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

E = Encoder(LATENT_DIM).cuda()
G = Generator(LATENT_DIM).cuda()
D = Discriminator().cuda()
C = CodeDiscriminator(LATENT_DIM).cuda()

opt_EG = torch.optim.Adam(
    list(E.parameters()) + list(G.parameters()),
    lr=LR,
    betas=(0.5,0.999)
)

opt_D = torch.optim.Adam(
    D.parameters(),
    lr=LR,
    betas=(0.5,0.999)
)

opt_C = torch.optim.Adam(
    C.parameters(),
    lr=LR,
    betas=(0.5,0.999)
)

trainer = Trainer(E,G,D,C,opt_EG,opt_D,opt_C)

for epoch in range(EPOCHS):

    for x in loader:

        x = x.cuda()

        loss = trainer.train_step(x)

    print("epoch",epoch,"loss",loss)