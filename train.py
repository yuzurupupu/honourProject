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
from evaluation.evaluate import evaluate

def main():

    device = torch.device(DEVICE)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = BraTSDataset("processed_brats2021/train")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    E = Encoder(LATENT_DIM).to(device)

    G = Generator(LATENT_DIM).to(device)

    D = Discriminator().to(device)

    C = CodeDiscriminator(LATENT_DIM).to(device)

    models = {
        "E": E,
        "G": G,
        "D": D,
        "C": C
    }
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")

        E = torch.nn.DataParallel(E)
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        C = torch.nn.DataParallel(C)

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------

    opt_EG = torch.optim.Adam(
        list(E.parameters()) + list(G.parameters()),
        lr=LR,
        betas=(0.5, 0.999)
    )

    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=LR,
        betas=(0.5, 0.999)
    )

    opt_C = torch.optim.Adam(
        C.parameters(),
        lr=LR,
        betas=(0.5, 0.999)
    )

    optimizers = {
        "EG": opt_EG,
        "D": opt_D,
        "C": opt_C
    }

    # --------------------------------------------------
    # Logger
    # --------------------------------------------------

    logger = Logger("runs")

    trainer = Trainer(models, optimizers, logger, device)

    step = 0

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        for batch_idx, x in enumerate(loader):

            losses = trainer.train_step(x, step)

            step += 1

            if batch_idx % 10 == 0:

                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"D {losses['loss_D']:.3f} | "
                    f"G {losses['loss_EG']:.3f}"
                )

        # --------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------

        if epoch % 10 == 0:

            save_checkpoint(
                models,
                optimizers,
                epoch,
                f"checkpoints/model_{epoch}.pth"
            )

        if epoch % 20 == 0:
            score = evaluate(G, loader, device)

            print("MS-SSIM:", score)

        # --------------------------------------------------
        # Generate sample
        # --------------------------------------------------

        if epoch % 5 == 0:

            z = torch.randn(1, LATENT_DIM).to(device)

            fake = G(z)

            volume = fake[0, 0].detach().cpu().numpy()

            save_slice(volume, f"samples/sample_epoch_{epoch}.png")


if __name__ == "__main__":

    main()