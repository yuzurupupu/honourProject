import torch
from pytorch_msssim import ms_ssim

def compute_msssim(real, fake):

    return ms_ssim(real, fake, data_range=1)