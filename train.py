import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader

from configs import *
from dataset.brats2021 import BraTSDataset
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
from models.code_discriminator import CodeDiscriminator
from training.trainer import Trainer
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from utils.visualizer import save_slice


def main():
    device = torch.device(DEVICE)

    dataset = BraTSDataset("processed_brats2021/train", modality="flair")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    E = Encoder(LATENT_DIM).to(device)
    G = Generator(LATENT_DIM).to(device)
    D = Discriminator().to(device)
    C = CodeDiscriminator(LATENT_DIM).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        E = torch.nn.DataParallel(E)
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        C = torch.nn.DataParallel(C)

    models = {
        "E": E,
        "G": G,
        "D": D,
        "C": C
    }

    opt_EG = torch.optim.Adam(
        list(E.parameters()) + list(G.parameters()),
        lr=LR,
        betas=(0.5, 0.9)
    )
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=LR, betas=(0.5, 0.9))

    optimizers = {
        "EG": opt_EG,
        "D": opt_D,
        "C": opt_C
    }

    logger = Logger("runs")
    trainer = Trainer(models, optimizers, logger, device)

    step = 0

    for epoch in range(EPOCHS):
        for batch_idx, x in enumerate(loader):
            losses = trainer.train_step(x, step)
            step += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"D {losses['loss_D']:.4f} | "
                    f"C {losses['loss_C']:.4f} | "
                    f"EG {losses['loss_EG']:.4f} | "
                    f"Recon {losses['recon']:.4f} | "
                    f"GP_D {losses['gp_D']:.4f} | "
                    f"GP_C {losses['gp_C']:.4f}"
                )

        if epoch % 10 == 0:
            save_checkpoint(models, optimizers, epoch, f"checkpoints/model_{epoch}.pth")

        if epoch % 10 == 0:
            z = torch.randn(1, LATENT_DIM).to(device)
            fake = G(z)
            volume = fake[0, 0].detach().cpu().numpy()
            save_slice(volume, f"samples/sample_epoch_{epoch}.png")


if __name__ == "__main__":
    main()