import torch
import torch.nn.functional as F
from losses import gradient_penalty
from configs import LATENT_DIM, LAMBDA_GP


class Trainer:

    def __init__(self,E,G,D,C,opt_EG,opt_D,opt_C):

        self.E = E
        self.G = G
        self.D = D
        self.C = C

        self.opt_EG = opt_EG
        self.opt_D = opt_D
        self.opt_C = opt_C


    def train_step(self,x):

        batch = x.size(0)

        z = torch.randn(batch,LATENT_DIM).cuda()

        z_hat = self.E(x)

        x_recon = self.G(z_hat)

        x_fake = self.G(z)

        # -------- train D --------

        real_score = self.D(x)

        fake_score = self.D(x_fake.detach())

        gp = gradient_penalty(self.D,x,x_fake)

        loss_D = fake_score.mean() - real_score.mean() + LAMBDA_GP*gp

        self.opt_D.zero_grad()
        loss_D.backward()
        self.opt_D.step()

        # -------- train C --------

        real_z = torch.randn(batch,LATENT_DIM).cuda()

        fake_z = z_hat.detach()

        loss_C = self.C(fake_z).mean() - self.C(real_z).mean()

        self.opt_C.zero_grad()
        loss_C.backward()
        self.opt_C.step()

        # -------- train E + G --------

        recon_loss = F.l1_loss(x_recon,x)

        adv_loss = -self.D(x_fake).mean()

        loss_EG = adv_loss + recon_loss

        self.opt_EG.zero_grad()
        loss_EG.backward()
        self.opt_EG.step()

        return loss_EG.item()