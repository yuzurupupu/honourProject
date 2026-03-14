import torch
import torch.nn.functional as F

from training.losses import gradient_penalty
from configs import LATENT_DIM, LAMBDA_GP, RECON_LAMBDA


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
        x = x.to(self.device)
        batch = x.size(0)

        # ---------------------------
        # 1) Encoder + Generator
        # original paper: update twice
        # ---------------------------
        eg_loss_value = 0.0
        recon_loss_value = 0.0

        for _ in range(2):
            ze = self.E(x)
            x_rec = self.G(ze)

            zr = torch.randn(batch, LATENT_DIM, device=self.device)
            x_rand = self.G(zr)

            loss_g = -0.5 * (self.D(x_rec).mean() + self.D(x_rand).mean())
            loss_e = -self.C(ze).mean()
            recon_loss = F.l1_loss(x_rec, x)

            loss_eg = loss_g + 0.1 * loss_e + RECON_LAMBDA * recon_loss

            self.opt_EG.zero_grad(set_to_none=True)
            loss_eg.backward()
            self.opt_EG.step()

            eg_loss_value += loss_eg.item()
            recon_loss_value += recon_loss.item()

        eg_loss_value /= 2.0
        recon_loss_value /= 2.0

        # ---------------------------
        # 2) Discriminator
        # ---------------------------
        with torch.no_grad():
            ze = self.E(x)
            x_rec = self.G(ze)
            zr = torch.randn(batch, LATENT_DIM, device=self.device)
            x_rand = self.G(zr)

        gp_d_real = torch.cat([x, x], dim=0)
        gp_d_fake = torch.cat([x_rec, x_rand], dim=0)
        gp_d = gradient_penalty(self.D, gp_d_real, gp_d_fake)

        loss_d = (
            self.D(x_rec).mean()
            + self.D(x_rand).mean()
            - 2.0 * self.D(x).mean()
            + LAMBDA_GP * gp_d
        )

        self.opt_D.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_D.step()

        # ---------------------------
        # 3) Code Discriminator
        # ---------------------------
        with torch.no_grad():
            ze = self.E(x)

        zr = torch.randn(batch, LATENT_DIM, device=self.device)
        gp_c = gradient_penalty(self.C, zr, ze)

        loss_c = self.C(ze).mean() - self.C(zr).mean() + 5.0 * gp_c

        self.opt_C.zero_grad(set_to_none=True)
        loss_c.backward()
        self.opt_C.step()

        if self.logger is not None:
            self.logger.log_loss("loss_D", loss_d.item(), step)
            self.logger.log_loss("loss_C", loss_c.item(), step)
            self.logger.log_loss("loss_EG", eg_loss_value, step)
            self.logger.log_loss("recon_loss", recon_loss_value, step)
            self.logger.log_loss("gp_D", gp_d.item(), step)
            self.logger.log_loss("gp_C", gp_c.item(), step)

        return {
            "loss_D": loss_d.item(),
            "loss_C": loss_c.item(),
            "loss_EG": eg_loss_value,
            "recon": recon_loss_value,
            "gp_D": gp_d.item(),
            "gp_C": gp_c.item(),
        }