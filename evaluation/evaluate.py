import torch
from evaluation.msssim import compute_msssim


def evaluate(G, loader, device):

    G.eval()

    scores = []

    with torch.no_grad():

        for x in loader:

            x = x.to(device)

            z = torch.randn(x.size(0),128).to(device)

            fake = G(z)

            score = compute_msssim(x, fake)

            scores.append(score.item())

    G.train()

    return sum(scores) / len(scores)