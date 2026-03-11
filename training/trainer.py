import torch
import torch.nn.functional as F

from training.losses import gradient_penalty
from configs import LATENT_DIM, LAMBDA_GP


class Trainer:

    def __init__(self, models, optimizers, logger, device):

        self.device = device

        self.E = models["E"]
        self.G = models["G"]
        self.D = models["D"]
        self.C = models["C"]

        self.opt_EG = optimizers["EG"]
        self.opt_D = optimizers["D"]
        self.opt_C = optimizers["C"]

        self.logger = logger


    def train_step(self, x, step):

        batch = x.size(0)

        x = x.to(self.device)

        # ------------------------------------------------
        # Encode
        # ------------------------------------------------

        z_hat = self.E(x)

        # reconstruction

        x_recon = self.G(z_hat)

        # random latent

        z = torch.randn(batch, LATENT_DIM).to(self.device)

        x_fake = self.G(z)

        # =================================================
        # Train Discriminator
        # =================================================

        real_score = self.D(x)

        fake_score = self.D(x_fake.detach())

        gp = gradient_penalty(self.D, x, x_fake.detach())

        loss_D = fake_score.mean() - real_score.mean() + LAMBDA_GP * gp

        self.opt_D.zero_grad()

        loss_D.backward()

        self.opt_D.step()

        # =================================================
        # Train Code Discriminator
        # =================================================

        z_real = torch.randn(batch, LATENT_DIM).to(self.device)

        z_fake = z_hat.detach()

        real_z_score = self.C(z_real)

        fake_z_score = self.C(z_fake)

        loss_C = fake_z_score.mean() - real_z_score.mean()

        self.opt_C.zero_grad()

        loss_C.backward()

        self.opt_C.step()

        # =================================================
        # Train Encoder + Generator
        # =================================================

        adv_loss = -self.D(x_fake).mean()

        recon_loss = F.l1_loss(x_recon, x)

        code_loss = -self.C(z_hat).mean()

        loss_EG = adv_loss + recon_loss + 0.1 * code_loss

        self.opt_EG.zero_grad()

        loss_EG.backward()

        self.opt_EG.step()

        # =================================================
        # Logging
        # =================================================

        self.logger.log_loss("loss_D", loss_D.item(), step)
        self.logger.log_loss("loss_EG", loss_EG.item(), step)
        self.logger.log_loss("recon_loss", recon_loss.item(), step)

        return {
            "loss_D": loss_D.item(),
            "loss_EG": loss_EG.item(),
            "recon": recon_loss.item()
        }